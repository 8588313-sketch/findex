from __future__ import annotations

import json
import logging
import sys
import tracemalloc
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
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
from findex.builder import build_index
from findex.ranking import ScorerName, make_scorer, search
from findex.stats import compute_stats
from findex.storage import StorageFormat, open_index, save_index

app = typer.Typer(
    name="findex",
    help="Fast streaming inverted index and ranked Boolean/phrase search engine.",
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


@app.command("index")
def cmd_index(
    root: Annotated[Path, typer.Argument(help="Directory with .txt documents")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output index file path")] = Path("index.bin"),
    fmt: Annotated[StorageFormat | None, typer.Option("--format", help="Index format: bin or pickle")] = None,
    no_positions: Annotated[bool, typer.Option("--no-positions", help="Disable storing token positions")] = False,
) -> None:
    """Build and serialize an inverted index from a corpus directory."""
    resolved_fmt: StorageFormat = fmt if fmt is not None else (
        "pickle" if out.suffix in {".pkl", ".pickle"} else "bin"
    )
    with_positions = not no_positions

    try:
        if not root.exists():
            raise FileNotFoundError(f"Corpus directory not found: {root}")
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
            task_id = progress.add_task("Indexing corpus...", total=total_files)
            idx = build_index(
                root,
                with_positions=with_positions,
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
    log.info(
        "Built %r | postings=%d | file=%.1f KiB | peak=%.2f MiB",
        idx,
        total_postings,
        file_kib,
        peak / 1024 / 1024,
    )
    out_console.print(
        f"[green]Indexed[/green] {idx.num_docs} docs ({len(idx)} terms, {total_postings} postings) "
        f"-> [bold]{out}[/bold] ({file_kib:.1f} KiB, peak {peak / 1024 / 1024:.2f} MiB)"
    )


@app.command("search")
def cmd_search(
    index_path: Annotated[Path, typer.Argument(help="Path to saved index file")],
    query: Annotated[str, typer.Argument(help="Boolean or phrase query")],
    scorer: Annotated[ScorerName, typer.Option("--scorer", "-s", help="Ranking model: bm25 or tfidf")] = "bm25",
    top_k: Annotated[int, typer.Option("--top-k", "-k", min=1, help="Number of results to display")] = 5,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON array to stdout")] = False,
    repeat: Annotated[bool, typer.Option("--repeat", help="Run query twice to show @lru_cache HIT")] = False,
) -> None:
    """Search a saved inverted index and rank matching documents."""
    active_scorer = make_scorer(scorer)

    try:
        tracemalloc.start()
        with open_index(index_path) as idx:
            results = search(idx, query, scorer=active_scorer, top_k=top_k)
            if repeat:
                _ = search(idx, query, scorer=active_scorer, top_k=top_k)
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
    root: Annotated[Path, typer.Argument(help="Directory with .txt documents")],
    limit: Annotated[int | None, typer.Option("--limit", help="Limit number of documents")] = None,
    top: Annotated[int, typer.Option("--top", min=1, help="Top-N frequent terms")] = 10,
) -> None:
    """Compute streaming corpus vocabulary statistics."""
    try:
        if not root.exists():
            raise FileNotFoundError(f"Corpus directory not found: {root}")
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


if __name__ == "__main__":
    app()
