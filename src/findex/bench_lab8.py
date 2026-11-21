
"""Lab 8 benchmarks: NumPy vectorized BM25 speedup, flamegraph_after.svg, and Hybrid RRF Precision@5.

Usage: uv run python -m findex.bench_lab8 data/
"""
from __future__ import annotations

import argparse
import cProfile
import logging
import pstats
import time
from pathlib import Path
from typing import Any, cast

from findex.builder import build_index
from findex.models import Index, SearchResult
from findex.ranking import (
    BM25,
    TfIdf,
    clear_query_cache,
    search,
    search_python_baseline,
)
from findex.semantic import (
    SemanticIndex,
    build_embeddings,
    default_semantic_dir,
    hybrid_search,
    load_semantic_index,
    semantic_search,
)
from findex.storage import load_index, save_index

log = logging.getLogger("findex.bench_lab8")


def benchmark_numpy_speedup(index: Index, iterations: int = 2000) -> list[tuple[str, str, float, float, float]]:
    """Compare Lab 3 pure-Python scorer vs Lab 8 NumPy vectorized scorer + np.argpartition."""
    index.prewarm_numpy()
    sorted_terms = sorted(
        [(t, len(pl)) for t, pl in index.items() if t.isalpha() and len(t) >= 4],
        key=lambda kv: kv[1],
    )
    rare_term = next((t for t, df in sorted_terms if df == 1), sorted_terms[0][0])
    freq_term = sorted_terms[-1][0]
    top3_terms = [t for t, _ in sorted_terms[-3:]]
    three_term_query = " OR ".join(top3_terms)

    cases = [
        ("Frequent term", freq_term),
        ("Rare term", rare_term),
        ("3-term OR query", three_term_query),
    ]

    bm25 = BM25()
    tfidf = TfIdf()
    rows: list[tuple[str, str, float, float, float]] = []

    for label, q in cases:
        # Verify exact equivalence of top-10 doc_ids & scores between Lab 3 and Lab 8
        ref_bm25 = search_python_baseline(index, q, scorer=bm25, top_k=10, include_snippets=False)
        vec_bm25 = search(index, q, scorer=bm25, top_k=10, include_snippets=False)
        assert [r.doc_id for r in ref_bm25] == [r.doc_id for r in vec_bm25], f"Mismatch on {q}"
        for r1, r2 in zip(ref_bm25, vec_bm25):
            assert abs(r1.score - r2.score) < 1e-9

        ref_tf = search_python_baseline(index, q, scorer=tfidf, top_k=10, include_snippets=False)
        vec_tf = search(index, q, scorer=tfidf, top_k=10, include_snippets=False)
        assert [r.doc_id for r in ref_tf] == [r.doc_id for r in vec_tf]

        # Warmup
        for _ in range(50):
            _ = search_python_baseline(index, q, scorer=bm25, top_k=10, include_snippets=False)
            _ = search(index, q, scorer=bm25, top_k=10, include_snippets=False)

        t0 = time.perf_counter()
        for _ in range(iterations):
            _ = search_python_baseline(index, q, scorer=bm25, top_k=10, include_snippets=False)
        ms_lab3 = ((time.perf_counter() - t0) / iterations) * 1000.0

        t1 = time.perf_counter()
        for _ in range(iterations):
            _ = search(index, q, scorer=bm25, top_k=10, include_snippets=False)
        ms_numpy = ((time.perf_counter() - t1) / iterations) * 1000.0

        speedup = ms_lab3 / ms_numpy if ms_numpy > 0 else 0.0
        rows.append((label, q, ms_lab3, ms_numpy, speedup))

    return rows


def generate_after_flamegraph(index: Index, out_svg: Path) -> None:
    """Capture cProfile stack distribution after NumPy vectorization for visual comparison."""
    index.prewarm_numpy()
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(4000):
        _ = search(index, "python OR module OR async", top_k=10, include_snippets=False)
    pr.disable()

    stats = cast(dict[tuple[str, int, str], tuple[int, int, float, float, dict[Any, Any]]], pstats.Stats(pr).stats)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
    rows: list[tuple[str, float, float, int]] = []
    for (fn, line, name), (_, nc, tt, ct, _) in sorted(stats.items(), key=lambda x: -x[1][3])[:22]:
        short_fn = Path(fn).name
        rows.append((f"{short_fn}:{line} ({name})", ct, tt, nc))

    max_ct = max((r[1] for r in rows), default=1.0) or 1.0
    w, bar_h = 980, 24
    h = 60 + len(rows) * (bar_h + 6)
    svg = [
        (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'style="font-family:monospace;background:#111827;color:#f9fafb">'),
        ('<text x="16" y="32" fill="#f9fafb" font-size="16" font-weight="bold">'
        "findex search() Flame Graph (After NumPy — 4000 x 3-term queries)</text>"),
    ]
    for i, (label, ct, tt, nc) in enumerate(rows):
        y = 50 + i * (bar_h + 6)
        bw = max(6, int((ct / max_ct) * (w - 32)))
        color = (
            "#10b981"
            if ("ranking.py" in label or "multiarray" in label or "numpy" in label)
            else ("#f59e0b" if "dataclasses" in label else "#3b82f6")
        )
        svg.append(f'<rect x="16" y="{y}" width="{bw}" height="{bar_h}" rx="4" fill="{color}" opacity="0.88"/>')
        svg.append(
            f'<text x="24" y="{y + 16}" fill="#ffffff" font-size="12">'
            f"{label} — cum={ct * 1000:.1f}ms, self={tt * 1000:.1f}ms, calls={nc}</text>"
        )
    svg.append("</svg>")
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    out_svg.write_text("\n".join(svg), encoding="utf-8")


def evaluate_precision_at_5(index: Index, sem_index: SemanticIndex) -> None:
    """Evaluate Precision@5 on 10 Lab 3 queries + 5 paraphrased queries without exact keywords."""
    lab3_queries: list[tuple[str, tuple[str, ...]]] = [
        ("київ AND столиця", ("Київ", "Історія_України")),
        ("тарас AND шевченко", ("Тарас_Шевченко", "Список_пам")),
        ('"повномасштабне вторгнення" OR "російського вторгнення"', ("Хронологія_російського_вторгнення", "Удари_по_об")),
        ("театр AND (2019 OR 2020)", ("2019_у_театрі", "2020_у_театрі", "Театр_під_час_війни")),
        ("львів AND карпати", ("Львів", "Карпати", "Мшанець")),
        ("приватбанк AND націоналізація", ("Націоналізація_ПриватБанку",)),
        ("євромайдан AND революція", ("Євромайдан", "Історія_України")),
        ("кримське AND ханство", ("Кримське_ханство", "Історія_Росії", "Історія_України")),
        ("python AND (programming OR language)", ("Python",)),
        ("alan AND turing", ("Alan_Turing",)),
    ]

    paraphrased_queries: list[tuple[str, tuple[str, ...]]] = [
        ("first female computer programmer analytical engine", ("Ada_Lovelace",)),
        ("apollo lunar landing craters orbit earth", ("Moon",)),
        ("open source operating system kernel torvalds", ("Linux",)),
        ("місто лева культурна столиця заходу україни", ("Львів",)),
        ("переведення найбільшого комерційного банку у державну власність", ("Націоналізація_ПриватБанку",)),
    ]

    def calc_p5(results: list[SearchResult], markers: tuple[str, ...]) -> float:
        if not results:
            return 0.0
        hits = sum(1 for r in results if any(m in r.path for m in markers))
        return hits / min(5, max(len(markers), 1))

    bm25 = BM25()
    log.info("\n=== PRECISION@5: 10 LAB-3 QUERIES + 5 PARAPHRASED QUERIES ===")
    log.info("%-58s | %6s | %8s | %11s", "Query", "BM25", "Semantic", "Hybrid(RRF)")
    log.info("-" * 93)

    all_bm25: list[float] = []
    all_sem: list[float] = []
    all_hyb: list[float] = []
    para_bm25: list[float] = []
    para_sem: list[float] = []
    para_hyb: list[float] = []

    for idx_q, (q, markers) in enumerate(lab3_queries + paraphrased_queries):
        r_bm = search(index, q, scorer=bm25, top_k=5, include_snippets=False)
        r_se = semantic_search(index, sem_index, q, top_k=5, include_snippets=False)
        r_hy = hybrid_search(index, sem_index, q, scorer=bm25, top_k=5, include_snippets=False)

        p_bm = calc_p5(r_bm, markers)
        p_se = calc_p5(r_se, markers)
        p_hy = calc_p5(r_hy, markers)

        all_bm25.append(p_bm)
        all_sem.append(p_se)
        all_hyb.append(p_hy)
        if idx_q >= len(lab3_queries):
            para_bm25.append(p_bm)
            para_sem.append(p_se)
            para_hyb.append(p_hy)

        tag = "[P]" if idx_q >= len(lab3_queries) else "[L3]"
        log.info("%-58s | %6.2f | %8.2f | %11.2f", f"{tag} {q[:52]}", p_bm, p_se, p_hy)

    log.info("-" * 93)
    log.info(
        "%-58s | %6.3f | %8.3f | %11.3f",
        "MEAN P@5 (All 15 queries)",
        sum(all_bm25) / len(all_bm25),
        sum(all_sem) / len(all_sem),
        sum(all_hyb) / len(all_hyb),
    )
    log.info(
        "%-58s | %6.3f | %8.3f | %11.3f",
        "MEAN P@5 (5 Paraphrased queries without exact words)",
        sum(para_bm25) / len(para_bm25),
        sum(para_sem) / len(para_sem),
        sum(para_hyb) / len(para_hyb),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Run Lab 8 benchmarks")
    parser.add_argument("root", type=Path, nargs="?", default=Path("data"))
    args = parser.parse_args()

    clear_query_cache()
    idx_path = args.root / "index.bin"
    if idx_path.exists():
        index = load_index(idx_path)
    else:
        index = build_index(args.root, with_positions=True)
        save_index(index, idx_path)

    log.info("=== 1. NUMPY VECTORIZED SCORING SPEEDUP (LAB 3 vs LAB 8) ===")
    rows = benchmark_numpy_speedup(index, iterations=1500)
    log.info("%-18s | %-28s | %10s | %10s | %8s", "Case", "Query", "ms_lab3", "ms_numpy", "speedup")
    log.info("-" * 86)
    for label, q, ms3, ms8, sp in rows:
        log.info("%-18s | %-28s | %10.4f | %10.4f | %7.2fx", label, q[:28], ms3, ms8, sp)

    generate_after_flamegraph(index, Path("docs/flamegraph_after.svg"))
    log.info("Saved docs/flamegraph_after.svg")

    sem_dir = default_semantic_dir(idx_path)
    if (sem_dir / "embeddings.npy").exists():
        sem_index = load_semantic_index(sem_dir)
    else:
        log.info("Building semantic embeddings into %s ...", sem_dir)
        sem_index = build_embeddings(index, out_dir=sem_dir)

    evaluate_precision_at_5(index, sem_index)


if __name__ == "__main__":
    main()
