from __future__ import annotations

import asyncio
import email.utils
import json
import logging
import random
import re
import time
from collections import defaultdict
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import httpx

DEFAULT_USER_AGENT = (
    "findex-crawler/0.6 (+https://github.com/8588313-sketch/findex; "
    "contact: 8588313-sketch@users.noreply.github.com)"
)

log = logging.getLogger("findex.crawler")


@dataclass(frozen=True, slots=True)
class FetchResult:
    """Outcome of a single URL fetch attempt (including retries)."""

    url: str
    final_url: str
    status_code: int | None
    body: str
    bytes_received: int
    elapsed_s: float
    attempts: int
    error: str | None = None


@dataclass(frozen=True, slots=True)
class CrawledPage:
    """Extracted document emitted incrementally by the async crawler."""

    url: str
    title: str
    text: str
    links: tuple[str, ...]
    status_code: int
    bytes_received: int
    elapsed_s: float


@dataclass(slots=True)
class CrawlStats:
    """Live telemetry counters updated while crawling."""

    pages_collected: int = 0
    errors: int = 0
    in_flight: int = 0
    queue_depth: int = 0
    bytes_total: int = 0
    started_at: float = field(default_factory=time.perf_counter)

    @property
    def elapsed_s(self) -> float:
        return max(1e-6, time.perf_counter() - self.started_at)

    @property
    def pages_per_sec(self) -> float:
        return self.pages_collected / self.elapsed_s


def canonicalize_url(raw_url: str, base_url: str | None = None) -> str | None:
    """Normalize URL: lowercase scheme/host, drop #fragment, normalize path & sorted query."""
    joined = urljoin(base_url, raw_url.strip()) if base_url else raw_url.strip()
    if not joined:
        return None
    parts = urlsplit(joined)
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"}:
        return None

    hostname = (parts.hostname or "").lower()
    if not hostname:
        return None

    port = parts.port
    if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
        port = None
    netloc = f"{hostname}:{port}" if port is not None else hostname

    # Collapse duplicate slashes and normalize trailing slash (preserve root '/')
    path = re.sub(r"/{2,}", "/", parts.path or "/")
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    # Sort query parameters deterministically
    query_pairs = sorted(parse_qsl(parts.query, keep_blank_values=True))
    query = urlencode(query_pairs)

    return urlunsplit((scheme, netloc, path, query, ""))


class _LinkAndTextExtractor(HTMLParser):
    """Lightweight stdlib HTML parser extracting visible text, <title>, and <a href> links."""

    _SKIP_TAGS = frozenset({"script", "style", "noscript", "svg", "template"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._in_title = False
        self.title_chunks: list[str] = []
        self.text_chunks: list[str] = []
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        if t in self._SKIP_TAGS:
            self._skip_depth += 1
        elif t == "title":
            self._in_title = True
        elif t == "a":
            for k, v in attrs:
                if k.lower() == "href" and v:
                    self.hrefs.append(v)

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif t == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        stripped = " ".join(data.split())
        if not stripped:
            return
        if self._in_title:
            self.title_chunks.append(stripped)
        self.text_chunks.append(stripped)


def extract_links_and_text(html: str, base_url: str) -> tuple[str, str, tuple[str, ...]]:
    """Parse HTML into (title, clean_text, canonical_links)."""
    parser = _LinkAndTextExtractor()
    parser.feed(html)
    parser.close()

    title = " ".join(parser.title_chunks).strip() or base_url
    text = " ".join(parser.text_chunks).strip()

    seen_links: set[str] = set()
    ordered_links: list[str] = []
    for href in parser.hrefs:
        canon = canonicalize_url(href, base_url=base_url)
        if canon and canon not in seen_links:
            seen_links.add(canon)
            ordered_links.append(canon)
    return title, text, tuple(ordered_links)


class RobotsCache:
    """Fetches, parses, and caches robots.txt rules per host."""

    def __init__(self, user_agent: str = DEFAULT_USER_AGENT, timeout_s: float = 5.0) -> None:
        self.user_agent = user_agent
        self.timeout_s = timeout_s
        self._parsers: dict[str, RobotFileParser | None] = {}
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def can_fetch(self, client: httpx.AsyncClient, url: str) -> tuple[bool, float]:
        """Return (is_allowed, crawl_delay_seconds) for the given URL."""
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        lock = self._locks[origin]
        async with lock:
            if origin not in self._parsers:
                robots_url = f"{origin}/robots.txt"
                rp = RobotFileParser()
                rp.set_url(robots_url)
                try:
                    async with asyncio.timeout(self.timeout_s):
                        resp = await client.get(robots_url)
                    if resp.status_code == 200:
                        rp.parse(resp.text.splitlines())
                        self._parsers[origin] = rp
                    elif resp.status_code in {401, 403}:
                        rp.parse(["User-agent: *", "Disallow: /"])
                        self._parsers[origin] = rp
                    else:
                        # 404 or missing robots.txt -> allowed by standard convention
                        self._parsers[origin] = None
                except (httpx.HTTPError, TimeoutError):
                    self._parsers[origin] = None

        cached = self._parsers[origin]
        if cached is None:
            return True, 0.0
        allowed = cached.can_fetch(self.user_agent, url)
        delay_val = cached.crawl_delay(self.user_agent)
        delay = float(delay_val) if delay_val is not None else 0.0
        return allowed, delay


def _parse_retry_after(header_value: str | None) -> float | None:
    if not header_value:
        return None
    raw = header_value.strip()
    try:
        seconds = float(raw)
        return max(0.0, seconds)
    except ValueError:
        pass
    try:
        parsed_dt = email.utils.parsedate_to_datetime(raw)
        if parsed_dt.tzinfo is None:
            parsed_dt = parsed_dt.replace(tzinfo=UTC)
        delta = (parsed_dt - datetime.now(UTC)).total_seconds()
        return max(0.0, delta)
    except (TypeError, ValueError):
        return None


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout_s: float = 10.0,
    max_retries: int = 3,
    base_backoff_s: float = 0.25,
    sleep_fn: Callable[[float], asyncio.Future[None] | object] | None = None,
) -> FetchResult:
    """Fetch a single URL with asyncio.timeout, exponential backoff + jitter, and Retry-After.

    Retries ONLY on HTTP 429, 5xx, or timeouts/network errors (at most max_retries times).
    Never retries 4xx (other than 429) or redirect loops.
    """
    sleeper = sleep_fn if sleep_fn is not None else asyncio.sleep
    t0 = time.perf_counter()
    attempts = 0
    last_status: int | None = None
    last_error: str | None = None

    while True:
        attempts += 1
        try:
            async with asyncio.timeout(timeout_s):
                resp = await client.get(url)
            last_status = resp.status_code
            raw_bytes = resp.content
            elapsed = time.perf_counter() - t0

            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                last_error = f"HTTP {resp.status_code}"
                if attempts > max_retries:
                    return FetchResult(
                        url=url,
                        final_url=str(resp.url),
                        status_code=resp.status_code,
                        body="",
                        bytes_received=len(raw_bytes),
                        elapsed_s=elapsed,
                        attempts=attempts,
                        error=last_error,
                    )
                retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
                jitter = random.uniform(0.0, base_backoff_s * 0.5)
                delay = (
                    retry_after
                    if retry_after is not None
                    else (base_backoff_s * (2 ** (attempts - 1)) + jitter)
                )
                log.warning(
                    "RETRY %s | status=%d attempt=%d/%d delay=%.3fs",
                    url,
                    resp.status_code,
                    attempts,
                    max_retries,
                    delay,
                )
                await sleeper(delay)  # type: ignore[misc]
                continue

            if 400 <= resp.status_code < 500:
                return FetchResult(
                    url=url,
                    final_url=str(resp.url),
                    status_code=resp.status_code,
                    body="",
                    bytes_received=len(raw_bytes),
                    elapsed_s=elapsed,
                    attempts=attempts,
                    error=f"HTTP {resp.status_code} (no retry)",
                )

            return FetchResult(
                url=url,
                final_url=str(resp.url),
                status_code=resp.status_code,
                body=resp.text,
                bytes_received=len(raw_bytes),
                elapsed_s=elapsed,
                attempts=attempts,
                error=None,
            )

        except httpx.TooManyRedirects as exc:
            elapsed = time.perf_counter() - t0
            return FetchResult(
                url=url,
                final_url=url,
                status_code=None,
                body="",
                bytes_received=0,
                elapsed_s=elapsed,
                attempts=attempts,
                error=f"TooManyRedirects: {exc}",
            )
        except (TimeoutError, httpx.TimeoutException, httpx.TransportError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempts > max_retries:
                elapsed = time.perf_counter() - t0
                return FetchResult(
                    url=url,
                    final_url=url,
                    status_code=last_status,
                    body="",
                    bytes_received=0,
                    elapsed_s=elapsed,
                    attempts=attempts,
                    error=last_error,
                )
            jitter = random.uniform(0.0, base_backoff_s * 0.5)
            delay = base_backoff_s * (2 ** (attempts - 1)) + jitter
            log.warning(
                "RETRY %s | error=%s attempt=%d/%d delay=%.3fs",
                url,
                type(exc).__name__,
                attempts,
                max_retries,
                delay,
            )
            await sleeper(delay)  # type: ignore[misc]


async def crawl(
    start_urls: list[str],
    *,
    max_pages: int = 100,
    concurrency: int = 10,
    per_host_limit: int = 2,
    min_host_delay_s: float = 0.05,
    timeout_s: float = 10.0,
    max_retries: int = 3,
    base_backoff_s: float = 0.25,
    same_domain_only: bool = True,
    allowed_path_prefix: str | None = None,
    user_agent: str = DEFAULT_USER_AGENT,
    transport: httpx.AsyncBaseTransport | None = None,
    stats: CrawlStats | None = None,
    crawl_log_path: Path | None = None,
) -> AsyncIterator[CrawledPage]:
    """Async generator yielding CrawledPage objects as pages arrive over the network."""
    if max_pages < 1:
        return

    live_stats = stats if stats is not None else CrawlStats()
    queue: asyncio.Queue[str] = asyncio.Queue()
    out_queue: asyncio.Queue[CrawledPage | None] = asyncio.Queue()
    seen: set[str] = set()
    allowed_hosts: set[str] = set()

    for raw in start_urls:
        canon = canonicalize_url(raw)
        if canon and canon not in seen:
            seen.add(canon)
            allowed_hosts.add(urlsplit(canon).netloc)
            queue.put_nowait(canon)

    live_stats.queue_depth = queue.qsize()
    if queue.empty():
        return

    global_sem = asyncio.Semaphore(max(1, concurrency))
    host_sems: dict[str, asyncio.Semaphore] = defaultdict(
        lambda: asyncio.Semaphore(max(1, per_host_limit))
    )
    host_rate_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
    host_last_request: dict[str, float] = {}
    robots = RobotsCache(user_agent=user_agent, timeout_s=timeout_s)

    log_fh = crawl_log_path.open("w", encoding="utf-8") if crawl_log_path is not None else None

    def _write_log(line: str) -> None:
        if log_fh is not None:
            log_fh.write(line + "\n")
            log_fh.flush()

    stop_event = asyncio.Event()
    busy_workers = 0

    async def _respect_host_delay(host: str, robots_delay: float) -> None:
        required_delay = max(min_host_delay_s, robots_delay)
        if required_delay <= 0.0:
            return
        async with host_rate_locks[host]:
            now = time.perf_counter()
            last = host_last_request.get(host, 0.0)
            wait_for = required_delay - (now - last)
            if wait_for > 0.0:
                await asyncio.sleep(wait_for)
            host_last_request[host] = time.perf_counter()

    async def _worker(client: httpx.AsyncClient) -> None:
        nonlocal busy_workers
        while not stop_event.is_set():
            try:
                async with asyncio.timeout(0.1):
                    url = await queue.get()
            except TimeoutError:
                if busy_workers == 0 and queue.empty():
                    stop_event.set()
                    return
                continue

            busy_workers += 1
            live_stats.queue_depth = queue.qsize()
            if stop_event.is_set():
                busy_workers -= 1
                queue.task_done()
                return

            host = urlsplit(url).netloc
            try:
                async with global_sem, host_sems[host]:
                    if stop_event.is_set():
                        return
                    allowed, robots_delay = await robots.can_fetch(client, url)
                    if not allowed:
                        live_stats.errors += 1
                        _write_log(f"DISALLOWED\t{url}\trobots.txt")
                        continue

                    await _respect_host_delay(host, robots_delay)
                    if stop_event.is_set():
                        return

                    live_stats.in_flight += 1
                    try:
                        res = await fetch(
                            client,
                            url,
                            timeout_s=timeout_s,
                            max_retries=max_retries,
                            base_backoff_s=base_backoff_s,
                        )
                    finally:
                        live_stats.in_flight -= 1

                    if res.error is not None or res.status_code is None:
                        live_stats.errors += 1
                        _write_log(
                            f"ERROR\t{url}\tstatus={res.status_code}\tattempts={res.attempts}\t"
                            f"time={res.elapsed_s:.3f}s\t{res.error}"
                        )
                        continue

                    title, text, links = extract_links_and_text(res.body, res.final_url)
                    page = CrawledPage(
                        url=res.final_url,
                        title=title,
                        text=text,
                        links=links,
                        status_code=res.status_code,
                        bytes_received=res.bytes_received,
                        elapsed_s=res.elapsed_s,
                    )

                    if live_stats.pages_collected >= max_pages:
                        stop_event.set()
                        return

                    live_stats.pages_collected += 1
                    live_stats.bytes_total += res.bytes_received
                    _write_log(
                        f"OK\t{res.final_url}\tstatus={res.status_code}\tbytes={res.bytes_received}\t"
                        f"attempts={res.attempts}\ttime={res.elapsed_s:.3f}s"
                    )
                    await out_queue.put(page)

                    if live_stats.pages_collected >= max_pages:
                        stop_event.set()
                        return

                    for link in links:
                        parsed_link = urlsplit(link)
                        if same_domain_only and parsed_link.netloc not in allowed_hosts:
                            continue
                        if allowed_path_prefix and not parsed_link.path.startswith(
                            allowed_path_prefix
                        ):
                            continue
                        if link not in seen and len(seen) < max_pages * 15:
                            seen.add(link)
                            queue.put_nowait(link)
                    live_stats.queue_depth = queue.qsize()
            finally:
                busy_workers -= 1
                queue.task_done()

    async def _run_pool() -> None:
        headers = {"User-Agent": user_agent}
        try:
            async with httpx.AsyncClient(
                headers=headers,
                follow_redirects=True,
                max_redirects=5,
                transport=transport,
            ) as client:
                try:
                    async with asyncio.TaskGroup() as tg:
                        workers = [tg.create_task(_worker(client)) for _ in range(max(1, concurrency))]
                        await stop_event.wait()
                        for w in workers:
                            w.cancel()
                except* asyncio.CancelledError:
                    pass
        finally:
            await out_queue.put(None)
            if log_fh is not None:
                log_fh.close()

    pool_task = asyncio.create_task(_run_pool())
    try:
        while True:
            item = await out_queue.get()
            if item is None:
                break
            yield item
    finally:
        stop_event.set()
        await pool_task


async def crawl_to_jsonl(
    start_urls: list[str],
    out_jsonl: Path,
    *,
    max_pages: int = 100,
    concurrency: int = 10,
    per_host_limit: int = 2,
    min_host_delay_s: float = 0.05,
    timeout_s: float = 10.0,
    max_retries: int = 3,
    base_backoff_s: float = 0.25,
    same_domain_only: bool = True,
    allowed_path_prefix: str | None = None,
    user_agent: str = DEFAULT_USER_AGENT,
    transport: httpx.AsyncBaseTransport | None = None,
    stats: CrawlStats | None = None,
    crawl_log_path: Path | None = None,
    on_page: Callable[[CrawledPage], None] | None = None,
) -> CrawlStats:
    """Stream crawled pages directly to a JSONL file without holding the corpus in memory."""
    live_stats = stats if stats is not None else CrawlStats()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as fh:
        async for page in crawl(
            start_urls,
            max_pages=max_pages,
            concurrency=concurrency,
            per_host_limit=per_host_limit,
            min_host_delay_s=min_host_delay_s,
            timeout_s=timeout_s,
            max_retries=max_retries,
            base_backoff_s=base_backoff_s,
            same_domain_only=same_domain_only,
            allowed_path_prefix=allowed_path_prefix,
            user_agent=user_agent,
            transport=transport,
            stats=live_stats,
            crawl_log_path=crawl_log_path,
        ):
            record = {
                "url": page.url,
                "title": page.title,
                "text": page.text,
                "status_code": page.status_code,
                "bytes": page.bytes_received,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            if on_page is not None:
                on_page(page)
    return live_stats
