from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tracemalloc
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.live import Live
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from findex.ast_query import QuerySyntaxError
from findex.builder import ExecutorKind, build_index_parallel
from findex.crawler import DEFAULT_USER_AGENT, CrawlStats, crawl_to_jsonl
from findex.ranking import ScorerName, make_scorer, search
from findex.semantic import (
    DEFAULT_EMBED_MODEL,
    SearchMode,
    build_embeddings,
    default_semantic_dir,
    hybrid_search,
    load_semantic_index,
    semantic_search,
)
from findex.stats import compute_stats
from findex.storage import StorageFormat, open_index, save_index

app = typer.Typer(
    name="findex",
    help="Fast streaming inverted index, ranked Boolean/phrase search engine, and async crawler.",
    no_args_is_help=True,
    add_completion=False,
)

err_console = Console(stderr=True)
out_console = Console()
log = logging.getLogger("findex.cli")


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)

    handler = RichHandler(
        console=err_console,
        show_time=False,
        show_path=verbosity >= 2,
        rich_tracebacks=False,
    )
    handler.setLevel(level)
    root_logger.addHandler(handler)


def _fail(message: str) -> None:
    err_console.print(f"[bold red]Error:[/bold red] {message}")
    raise typer.Exit(code=1)


@app.callback()
def main_callback(
    verbose: Annotated[
        int,
        typer.Option(
            "-v",
            "--verbose",
            count=True,
            help="Increase log verbosity (-v for INFO, -vv for DEBUG).",
        ),
    ] = 0,
) -> None:
    _configure_logging(verbose)


def _render_crawl_status(stats: CrawlStats, max_pages: int, out: Path) -> Table:
    table = Table(title=f"Async Crawler -> {out}")
    table.add_column("Collected", justify="right", style="green")
    table.add_column("Pages/s", justify="right", style="cyan")
    table.add_column("In-Flight", justify="right", style="yellow")
    table.add_column("Errors", justify="right", style="red")
    table.add_column("Queue Depth", justify="right", style="magenta")
    table.add_row(
        f"{stats.pages_collected}/{max_pages}",
        f"{stats.pages_per_sec:.2f}",
        str(stats.in_flight),
        str(stats.errors),
        str(stats.queue_depth),
    )
    return table


@app.command("crawl")
def cmd_crawl(
    start_url: Annotated[str, typer.Argument(help="Seed URL to start crawling from")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output JSONL path")] = Path("crawled.jsonl"),
    log_file: Annotated[Path, typer.Option("--log", help="Per-URL crawl log path")] = Path("crawl.log"),
    max_pages: Annotated[int, typer.Option("--max-pages", "-m", min=1, help="Maximum pages to collect")] = 50,
    concurrency: Annotated[int, typer.Option("--concurrency", "-c", min=1, help="Global concurrent requests")] = 10,
    per_host: Annotated[int, typer.Option("--per-host", min=1, help="Max concurrent requests per host")] = 2,
    delay: Annotated[float, typer.Option("--delay", min=0.0, help="Min seconds between requests to same host")] = 0.05,
    timeout: Annotated[float, typer.Option("--timeout", min=0.1, help="Per-request timeout in seconds")] = 10.0,
    path_prefix: Annotated[
        str | None,
        typer.Option("--path-prefix", help="Restrict crawling to URLs starting with this path prefix"),
    ] = None,
    debug_loop: Annotated[
        bool,
        typer.Option("--debug", help="Enable asyncio debug mode to detect slow callbacks blocking the loop"),
    ] = False,
) -> None:
    """Asynchronously crawl web pages and stream them into a JSONL file."""
    stats = CrawlStats()

    async def _run() -> None:
        if debug_loop:
            loop = asyncio.get_running_loop()
            loop.set_debug(True)
            loop.slow_callback_duration = 0.1

        with Live(_render_crawl_status(stats, max_pages, out), console=err_console, refresh_per_second=8) as live:
            await crawl_to_jsonl(
                [start_url],
                out,
                max_pages=max_pages,
                concurrency=concurrency,
                per_host_limit=per_host,
                min_host_delay_s=delay,
                timeout_s=timeout,
                allowed_path_prefix=path_prefix,
                user_agent=DEFAULT_USER_AGENT,
                stats=stats,
                crawl_log_path=log_file,
                on_page=lambda _p: live.update(_render_crawl_status(stats, max_pages, out)),
            )
            live.update(_render_crawl_status(stats, max_pages, out))

    try:
        asyncio.run(_run(), debug=debug_loop)
    except (OSError, ValueError) as exc:
        _fail(str(exc))
        return

    out_console.print(
        f"[green]Crawled[/green] {stats.pages_collected} pages in {stats.elapsed_s:.2f}s "
        f"({stats.pages_per_sec:.2f} pages/s, errors={stats.errors}) -> [bold]{out}[/bold] "
        f"(log: {log_file})"
    )


@app.command("index")
def cmd_index(
    root: Annotated[Path, typer.Argument(help="Directory with .txt documents or crawled .jsonl file")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output index file path")] = Path("index.bin"),
    fmt: Annotated[StorageFormat | None, typer.Option("--format", help="Index format: bin or pickle")] = None,
    no_positions: Annotated[bool, typer.Option("--no-positions", help="Disable storing token positions")] = False,
    workers: Annotated[
        int,
        typer.Option("--workers", "-w", min=1, help="Number of worker threads or processes"),
    ] = max(1, os.cpu_count() or 1),
    executor: Annotated[
        ExecutorKind,
        typer.Option("--executor", "-e", help="Execution backend: serial, threads, or processes"),
    ] = "processes",
) -> None:
    """Build and serialize an inverted index from a directory or .jsonl crawl output."""
    resolved_fmt: StorageFormat = fmt if fmt is not None else (
        "pickle" if out.suffix in {".pkl", ".pickle"} else "bin"
    )
    with_positions = not no_positions

    try:
        if not root.exists():
            raise FileNotFoundError(f"Corpus path not found: {root}")
        if root.is_file() and root.suffix.lower() == ".jsonl":
            with root.open("r", encoding="utf-8") as fh:
                total_files = sum(1 for line in fh if line.strip())
        else:
            total_files = sum(1 for _ in root.rglob("*.txt"))

        tracemalloc.start()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} docs"),
            TimeElapsedColumn(),
            console=err_console,
            transient=True,
        ) as progress:
            task_id = progress.add_task(
                f"Indexing ({executor}, workers={workers})...",
                total=total_files,
            )
            idx, build_stats = build_index_parallel(
                root,
                with_positions=with_positions,
                workers=workers,
                executor=executor,
                on_doc=lambda _doc: progress.advance(task_id, 1),
            )
            save_index(idx, out, fmt=resolved_fmt)

        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    except (OSError, ValueError) as exc:
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        _fail(str(exc))
        return

    file_kib = out.stat().st_size / 1024.0
    total_postings = sum(len(pl) for pl in idx.values())
    out_console.print(
        f"[green]Indexed[/green] {idx.num_docs} docs ({len(idx)} terms, {total_postings} postings) "
        f"via [cyan]{executor}[/cyan] (workers={workers}, wall={build_stats.wall_time_s:.3f}s, "
        f"merge={build_stats.merge_time_s * 1000:.1f}ms) "
        f"-> [bold]{out}[/bold] ({file_kib:.1f} KiB, peak {peak / 1024 / 1024:.2f} MiB)"
    )


@app.command("search")
def cmd_search(
    index_path: Annotated[Path, typer.Argument(help="Path to saved index file")],
    query: Annotated[str, typer.Argument(help="Boolean or phrase query")],
    scorer: Annotated[ScorerName, typer.Option("--scorer", "-s", help="Ranking model: bm25 or tfidf")] = "bm25",
    mode: Annotated[
        SearchMode,
        typer.Option("--mode", "-m", help="Search mode: lexical, semantic, or hybrid"),
    ] = "lexical",
    semantic_dir: Annotated[
        Path | None,
        typer.Option("--semantic-dir", help="Directory with embeddings.npy (defaults to <index>_semantic)"),
    ] = None,
    top_k: Annotated[int, typer.Option("--top-k", "-k", min=1, help="Number of results to display")] = 5,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON array to stdout")] = False,
    repeat: Annotated[bool, typer.Option("--repeat", help="Run query twice to show @lru_cache HIT")] = False,
) -> None:
    """Search a saved inverted index using lexical, semantic, or hybrid (RRF) ranking."""
    active_scorer = make_scorer(scorer)

    try:
        tracemalloc.start()
        with open_index(index_path) as idx:
            if mode == "lexical":
                results = search(idx, query, scorer=active_scorer, top_k=top_k)
                if repeat:
                    _ = search(idx, query, scorer=active_scorer, top_k=top_k)
            else:
                sem_path = semantic_dir if semantic_dir is not None else default_semantic_dir(index_path)
                sem_idx = load_semantic_index(sem_path)
                if mode == "semantic":
                    results = semantic_search(idx, sem_idx, query, top_k=top_k)
                else:
                    results = hybrid_search(idx, sem_idx, query, scorer=active_scorer, top_k=top_k)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    except (OSError, QuerySyntaxError, ValueError) as exc:
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        _fail(str(exc))
        return

    log.info("Peak memory: %.2f MiB", peak / 1024 / 1024)

    if as_json:
        payload = [asdict(r) for r in results]
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        return

    if not results:
        out_console.print(f"[yellow]No documents matched:[/yellow] {query}")
        return

    table = Table(title=f"Search results for: {query} ({active_scorer!r})")
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Doc ID", justify="right", style="magenta", no_wrap=True)
    table.add_column("Score", justify="right", style="green", no_wrap=True)
    table.add_column("Title", style="bold")
    table.add_column("Snippet")

    for rank, res in enumerate(results, start=1):
        table.add_row(
            str(rank),
            str(res.doc_id),
            f"{res.score:.4f}",
            res.title,
            res.snippet,
        )

    out_console.print(table)


@app.command("stats")
def cmd_stats(
    root: Annotated[Path, typer.Argument(help="Directory with .txt documents or .jsonl file")],
    limit: Annotated[int | None, typer.Option("--limit", help="Limit number of documents")] = None,
    top: Annotated[int, typer.Option("--top", min=1, help="Top-N frequent terms")] = 10,
) -> None:
    """Compute streaming corpus vocabulary statistics."""
    try:
        if not root.exists():
            raise FileNotFoundError(f"Corpus path not found: {root}")
        tracemalloc.start()
        n_docs, counts = compute_stats(root, limit=limit)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    except OSError as exc:
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        _fail(str(exc))
        return

    log.info("Peak memory: %.2f MiB", peak / 1024 / 1024)

    table = Table(title=f"Corpus Stats: {root} (docs={n_docs}, tokens={counts.total()}, vocab={len(counts)})")
    table.add_column("Rank", justify="right", style="cyan")
    table.add_column("Term", style="bold")
    table.add_column("Count", justify="right", style="green")

    for rank, (term, count) in enumerate(counts.most_common(top), start=1):
        table.add_row(str(rank), term, str(count))

    out_console.print(table)






@app.command("embed")
def cmd_embed(
    index_path: Annotated[Path, typer.Argument(help="Path to saved index.bin file")],
    out_dir: Annotated[
        Path | None,
        typer.Option("--out-dir", "-o", help="Output directory for embeddings.npy (defaults to <index>_semantic)"),
    ] = None,
    model: Annotated[
        str,
        typer.Option("--model", help="Sentence embedding model name"),
    ] = DEFAULT_EMBED_MODEL,
    chunk_size: Annotated[int, typer.Option("--chunk-size", min=20, help="Words per chunk")] = 120,
    overlap: Annotated[int, typer.Option("--overlap", min=0, help="Overlapping words between chunks")] = 30,
) -> None:
    """Split indexed documents into chunks, compute normalized embeddings, and save to .npy."""
    target_dir = out_dir if out_dir is not None else default_semantic_dir(index_path)
    try:
        with open_index(index_path) as idx:
            sem_idx = build_embeddings(
                idx,
                out_dir=target_dir,
                model_name=model,
                chunk_words=chunk_size,
                overlap_words=overlap,
            )
    except (OSError, ValueError) as exc:
        _fail(str(exc))
        return

    out_console.print(
        f"[green]Built semantic embeddings[/green]: chunks={sem_idx.num_chunks}, "
        f"dim={sem_idx.dim}, model={sem_idx.model_name} -> [bold]{target_dir / 'embeddings.npy'}[/bold]"
    )


@app.command("serve")
def cmd_serve(
    host: Annotated[str, typer.Option("--host", help="Bind host address")] = "0.0.0.0",
    port: Annotated[int, typer.Option("--port", "-p", min=1, max=65535, help="Bind TCP port")] = 8000,
    workers: Annotated[int, typer.Option("--workers", "-w", min=1, help="Number of Uvicorn worker processes")] = 1,
    index_path: Annotated[
        Path | None,
        typer.Option("--index-path", "-i", help="Path to serialized index file (overrides INDEX_PATH)"),
    ] = None,
) -> None:
    """Start the FastAPI web search server via Uvicorn."""
    import uvicorn

    if index_path is not None:
        os.environ["INDEX_PATH"] = str(index_path)
    resolved_idx = os.getenv("INDEX_PATH", "index.bin")
    out_console.print(
        f"[green]Starting findex web server[/green] on [bold]http://{host}:{port}[/bold] "
        f"(workers={workers}, INDEX_PATH={resolved_idx})"
    )
    uvicorn.run(
        "findex.web.app:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
    )


if __name__ == "__main__":
    app()
