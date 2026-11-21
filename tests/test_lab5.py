from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from findex.builder import (
    build_index,
    build_index_parallel,
    build_partial,
    merge_partials,
)
from findex.cli import app

runner = CliRunner()


def test_merge_single_partial_matches_lab4_build_index(sample_corpus: Path) -> None:
    """merge_partials([build_partial(all_paths)]) must produce the exact same index as Lab 4."""
    expected = build_index(sample_corpus, with_positions=True, workers=1, executor="serial")
    all_paths = sorted(sample_corpus.rglob("*.txt"))
    reconstructed = merge_partials([build_partial(all_paths, start_doc_id=0, with_positions=True)])

    assert reconstructed.docs == expected.docs
    assert reconstructed.postings == expected.postings
    assert reconstructed.with_positions == expected.with_positions


@pytest.mark.parametrize("executor", ["serial", "threads", "processes"])
@pytest.mark.parametrize("workers", [1, 2, 4])
def test_parallel_executors_match_serial(
    sample_corpus: Path,
    executor: str,
    workers: int,
) -> None:
    serial_idx, _ = build_index_parallel(sample_corpus, with_positions=True, workers=1, executor="serial")
    par_idx, stats = build_index_parallel(
        sample_corpus,
        with_positions=True,
        workers=workers,
        executor=executor,  # type: ignore[arg-type]
    )

    assert par_idx.docs == serial_idx.docs
    assert par_idx.postings == serial_idx.postings
    assert stats.workers == workers


def test_worker_exception_propagates_without_hanging(tmp_path: Path) -> None:
    """If a worker fails on a missing/unreadable path, build_partial/merge raises immediately."""
    missing_file = tmp_path / "non_existent.txt"
    with pytest.raises(FileNotFoundError):
        build_partial([missing_file], start_doc_id=0, with_positions=True)


@pytest.mark.parametrize("executor", ["serial", "threads", "processes"])
def test_cli_index_supports_workers_and_executor(
    sample_corpus: Path,
    tmp_path: Path,
    executor: str,
) -> None:
    out_file = tmp_path / f"idx_{executor}.bin"
    res = runner.invoke(
        app,
        ["index", str(sample_corpus), "--out", str(out_file), "--workers", "2", "--executor", executor],
    )
    assert res.exit_code == 0, res.output
    assert out_file.exists()
