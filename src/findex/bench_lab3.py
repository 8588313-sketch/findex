"""Lab 3 experiments: ranking sanity checks, Precision@5 (TF-IDF vs BM25), @timed & @lru_cache.

Usage: python -m findex.bench_lab3 data/
"""
import argparse
import logging
from pathlib import Path

from findex.builder import build_index
from findex.models import DocMeta, Index, Posting, SearchResult
from findex.ranking import BM25, TfIdf, clear_query_cache, search

log = logging.getLogger("findex.bench_lab3")


def run_sanity_checks(corpus_index: Index) -> None:
    log.info("=== 1. RANKING SANITY CHECKS ===")

    sorted_terms = sorted(corpus_index.items(), key=lambda kv: len(kv[1]))
    rare_term, rare_plist = next((t, pl) for t, pl in sorted_terms if len(pl) == 1 and len(t) > 4)
    freq_term, freq_plist = max(corpus_index.items(), key=lambda kv: len(kv[1]))

    bm25 = BM25()
    tfidf = TfIdf()
    rare_doc = rare_plist[0]
    freq_doc = freq_plist[0]

    s_rare_bm25 = bm25.score(
        rare_doc.tf, len(rare_plist), corpus_index.docs[rare_doc.doc_id].length,
        corpus_index.num_docs, corpus_index.avg_doc_length,
    )
    s_freq_bm25 = bm25.score(
        freq_doc.tf, len(freq_plist), corpus_index.docs[freq_doc.doc_id].length,
        corpus_index.num_docs, corpus_index.avg_doc_length,
    )
    s_rare_tfidf = tfidf.score(
        rare_doc.tf, len(rare_plist), corpus_index.docs[rare_doc.doc_id].length,
        corpus_index.num_docs, corpus_index.avg_doc_length,
    )
    s_freq_tfidf = tfidf.score(
        freq_doc.tf, len(freq_plist), corpus_index.docs[freq_doc.doc_id].length,
        corpus_index.num_docs, corpus_index.avg_doc_length,
    )
    log.info("  1) Rare vs Frequent term:")
    log.info(
        "     Rare  %r (df=%d, tf=%d) -> BM25=%.4f, TF-IDF=%.4f",
        rare_term, len(rare_plist), rare_doc.tf, s_rare_bm25, s_rare_tfidf,
    )
    log.info(
        "     Freq  %r (df=%d, tf=%d) -> BM25=%.4f, TF-IDF=%.4f",
        freq_term, len(freq_plist), freq_doc.tf, s_freq_bm25, s_freq_tfidf,
    )
    assert s_rare_bm25 > s_freq_bm25 and s_rare_tfidf > s_freq_tfidf

    num_docs = corpus_index.num_docs
    avg_dl = corpus_index.avg_doc_length
    dl = int(avg_dl)
    df_sample = 10
    bm25_1 = bm25.score(1, df_sample, dl, num_docs, avg_dl)
    bm25_2 = bm25.score(2, df_sample, dl, num_docs, avg_dl)
    bm25_19 = bm25.score(19, df_sample, dl, num_docs, avg_dl)
    bm25_20 = bm25.score(20, df_sample, dl, num_docs, avg_dl)
    gain_1_to_2 = bm25_2 - bm25_1
    gain_19_to_20 = bm25_20 - bm25_19
    log.info("  2) TF Saturation (BM25, df=%d, doc_len=%d):", df_sample, dl)
    log.info("     tf=1 -> %.4f, tf=2 -> %.4f (gain +%.4f)", bm25_1, bm25_2, gain_1_to_2)
    log.info("     tf=19 -> %.4f, tf=20 -> %.4f (gain +%.4f)", bm25_19, bm25_20, gain_19_to_20)
    assert gain_19_to_20 < gain_1_to_2 * 0.05

    toy_index = Index(
        postings={"київ": [Posting(doc_id=0, tf=1), Posting(doc_id=1, tf=1)]},
        docs={
            0: DocMeta(doc_id=0, path="short.txt", title="Short doc", length=50),
            1: DocMeta(doc_id=1, path="long.txt", title="Very long doc", length=20000),
        },
    )
    res = search(toy_index, "київ", scorer=bm25, top_k=2, include_snippets=False)
    log.info("  3) Short vs Long doc (both tf=1):")
    log.info("     Short (len=50):  score=%.4f (%s)", res[0].score, res[0].title)
    log.info("     Long (len=20000): score=%.4f (%s)", res[1].score, res[1].title)
    assert res[0].doc_id == 0 and res[0].score > res[1].score


def run_precision_eval(index: Index) -> None:
    log.info("=== 2. PRECISION@5 EVALUATION (10 QUERIES: TF-IDF vs BM25) ===")
    benchmark_queries: list[tuple[str, tuple[str, ...]]] = [
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

    p5_tfidf_list: list[float] = []
    p5_bm25_list: list[float] = []

    for q, relevant_markers in benchmark_queries:
        res_tfidf = search(index, q, scorer=TfIdf(), top_k=5, include_snippets=False)
        res_bm25 = search(index, q, scorer=BM25(), top_k=5, include_snippets=False)

        def calc_p5(results: list[SearchResult], markers: tuple[str, ...] = relevant_markers) -> float:
            if not results:
                return 0.0
            hits = sum(
                1 for r in results
                if any(marker in r.path for marker in markers)
            )
            return hits / min(5, max(len(markers), 1))

        p_tf = calc_p5(res_tfidf)
        p_bm = calc_p5(res_bm25)
        p5_tfidf_list.append(p_tf)
        p5_bm25_list.append(p_bm)
        log.info("  %-55s | P@5 TF-IDF=%.2f | BM25=%.2f", q, p_tf, p_bm)

    mean_tfidf = sum(p5_tfidf_list) / len(p5_tfidf_list)
    mean_bm25 = sum(p5_bm25_list) / len(p5_bm25_list)
    log.info("  MEAN P@5: TF-IDF=%.3f | BM25=%.3f", mean_tfidf, mean_bm25)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Run Lab 3 benchmarks")
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    clear_query_cache()
    index = build_index(args.root, with_positions=True)
    run_sanity_checks(index)
    run_precision_eval(index)

    log.info("=== 3. @timed & @lru_cache TRACE ===")
    demo_q = "київ AND (україна OR львів) NOT росія"
    _ = search(index, demo_q, scorer=BM25(), top_k=3, include_snippets=False)
    _ = search(index, demo_q, scorer=BM25(), top_k=3, include_snippets=False)


if __name__ == "__main__":
    main()
