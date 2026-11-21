from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from findex.builder import build_index
from findex.cli import app
from findex.corpus import iter_documents
from findex.crawler import (
    CrawlStats,
    RobotsCache,
    canonicalize_url,
    crawl,
    crawl_to_jsonl,
    fetch,
)
from findex.ranking import search

runner = CliRunner()


def test_canonicalize_url_normalizes_scheme_host_fragment_slash_and_query() -> None:
    u1 = canonicalize_url("HTTPS://Example.COM:443/docs/page/?b=2&a=1#section")
    u2 = canonicalize_url("http://example.com:80/docs/page?a=1&b=2")
    assert u1 == "https://example.com/docs/page?a=1&b=2"
    assert u2 == "http://example.com/docs/page?a=1&b=2"
    assert canonicalize_url("mailto:user@example.com") is None
    assert canonicalize_url("#only-fragment", base_url="https://example.com/a") == "https://example.com/a"


def test_robots_cache_respects_disallow_and_caches_per_host() -> None:
    robots_requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal robots_requests
        if request.url.path == "/robots.txt":
            robots_requests += 1
            return httpx.Response(200, text="User-agent: *\nDisallow: /private/\n")
        return httpx.Response(200, text="<html><body>ok</body></html>")

    async def _run() -> tuple[bool, bool, int]:
        transport = httpx.MockTransport(handler)
        cache = RobotsCache()
        async with httpx.AsyncClient(transport=transport) as client:
            ok1, _ = await cache.can_fetch(client, "https://docs.example.org/public/intro")
            ok2, _ = await cache.can_fetch(client, "https://docs.example.org/private/secret")
            ok3, _ = await cache.can_fetch(client, "https://docs.example.org/public/guide")
        return ok1, ok2, ok3, robots_requests

    allowed_pub, allowed_priv, allowed_pub2, count = asyncio.run(_run())
    assert allowed_pub is True
    assert allowed_priv is False
    assert allowed_pub2 is True
    assert count == 1


def test_fetch_retries_429_with_retry_after_and_503_but_not_404() -> None:
    calls: dict[str, int] = {"/rate": 0, "/flaky": 0, "/missing": 0}
    recorded_delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        recorded_delays.append(delay)

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        calls[path] = calls.get(path, 0) + 1
        if path == "/rate":
            if calls[path] == 1:
                return httpx.Response(429, headers={"Retry-After": "1.5"}, text="Too Many Requests")
            return httpx.Response(200, text="<html><title>Recovered</title><body>Done</body></html>")
        if path == "/flaky":
            if calls[path] <= 2:
                return httpx.Response(503, text="Service Unavailable")
            return httpx.Response(200, text="<html><body>Healthy</body></html>")
        if path == "/missing":
            return httpx.Response(404, text="Not Found")
        return httpx.Response(200, text="ok")

    async def _run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            r429 = await fetch(client, "https://example.org/rate", max_retries=3, sleep_fn=fake_sleep)
            assert r429.status_code == 200
            assert r429.attempts == 2
            assert recorded_delays[0] == pytest.approx(1.5)

            r503 = await fetch(client, "https://example.org/flaky", max_retries=3, sleep_fn=fake_sleep)
            assert r503.status_code == 200
            assert r503.attempts == 3

            r404 = await fetch(client, "https://example.org/missing", max_retries=3, sleep_fn=fake_sleep)
            assert r404.status_code == 404
            assert r404.attempts == 1
            assert calls["/missing"] == 1

    asyncio.run(_run())


def test_fetch_handles_timeout_and_redirect_loop_without_hanging() -> None:
    timeout_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal timeout_calls
        if request.url.path == "/timeout":
            timeout_calls += 1
            raise httpx.ReadTimeout("simulated read timeout")
        if request.url.path == "/loop-a":
            return httpx.Response(302, headers={"Location": "https://example.org/loop-b"})
        if request.url.path == "/loop-b":
            return httpx.Response(302, headers={"Location": "https://example.org/loop-a"})
        return httpx.Response(200, text="ok")

    async def fake_sleep(_delay: float) -> None:
        return None

    async def _run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=transport,
            follow_redirects=True,
            max_redirects=3,
        ) as client:
            res_to = await fetch(
                client,
                "https://example.org/timeout",
                timeout_s=0.1,
                max_retries=2,
                sleep_fn=fake_sleep,
            )
            assert res_to.status_code is None
            assert res_to.attempts == 3  # initial + 2 retries
            assert res_to.error is not None and "ReadTimeout" in res_to.error

            res_loop = await fetch(
                client,
                "https://example.org/loop-a",
                max_retries=3,
                sleep_fn=fake_sleep,
            )
            assert res_loop.attempts == 1
            assert res_loop.error is not None and "TooManyRedirects" in res_loop.error

    asyncio.run(_run())


def test_crawl_generator_deduplicates_enforces_max_pages_and_indexes_jsonl(tmp_path: Path) -> None:
    pages = {
        "/robots.txt": "User-agent: *\nDisallow: /blocked\n",
        "/": (
            "<html><title>Home</title><body>Welcome to asyncio crawler "
            "<a href='/p1#top'>P1</a> <a href='/p1?x=1#dup'>P1 Duplicate</a> "
            "<a href='/p2'>P2</a> <a href='/blocked'>Blocked</a></body></html>"
        ),
        "/p1": (
            "<html><title>Page One</title><body>Event loop and coroutines in Python "
            "<a href='/'>Home</a> <a href='/p3'>P3</a></body></html>"
        ),
        "/p2": "<html><title>Page Two</title><body>Semaphore and TaskGroup concurrency <a href='/p3'>P3</a></body></html>",
        "/p3": "<html><title>Page Three</title><body>Exponential backoff with jitter</body></html>",
        "/blocked": "<html><title>Secret</title><body>Should never be fetched</body></html>",
    }

    fetched_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        fetched_paths.append(path)
        if path in pages:
            return httpx.Response(200, text=pages[path])
        return httpx.Response(404, text="Not found")

    jsonl_out = tmp_path / "site.jsonl"
    log_out = tmp_path / "crawl.log"

    async def _run() -> CrawlStats:
        transport = httpx.MockTransport(handler)
        stats = CrawlStats()
        await crawl_to_jsonl(
            ["https://docs.example.org/"],
            jsonl_out,
            max_pages=3,
            concurrency=4,
            per_host_limit=2,
            min_host_delay_s=0.0,
            transport=transport,
            stats=stats,
            crawl_log_path=log_out,
        )
        return stats

    stats = asyncio.run(_run())
    assert stats.pages_collected == 3
    assert "/blocked" not in fetched_paths
    assert log_out.exists()

    # Stream JSONL via Lab 1 iter_documents and index via Lab 4/5 build_index
    streamed_docs = list(iter_documents(jsonl_out))
    assert len(streamed_docs) == 3

    idx = build_index(jsonl_out, with_positions=True)
    hits = search(idx, "coroutines OR semaphore", top_k=5)
    assert len(hits) >= 1

    # Also verify CLI search works end-to-end on index built from JSONL
    idx_file = tmp_path / "crawled.bin"
    res_idx = runner.invoke(app, ["index", str(jsonl_out), "--out", str(idx_file), "--executor", "serial"])
    assert res_idx.exit_code == 0, res_idx.output

    res_search = runner.invoke(app, ["search", str(idx_file), "asyncio", "--json"])
    assert res_search.exit_code == 0, res_search.output
    assert "Home" in res_search.output


def test_crawl_async_generator_yields_incrementally() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(200, text="<html><title>Single</title><body>Only one page</body></html>")

    async def _collect() -> list[str]:
        transport = httpx.MockTransport(handler)
        titles: list[str] = []
        async for page in crawl(
            ["https://single.example.org/"],
            max_pages=5,
            concurrency=2,
            min_host_delay_s=0.0,
            transport=transport,
        ):
            titles.append(page.title)
        return titles

    assert asyncio.run(_collect()) == ["Single"]
