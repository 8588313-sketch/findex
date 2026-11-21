"""Lab 3 experiments: ranking sanity checks, Precision@5 (TF-IDF vs BM25), @timed & @lru_cache.

Usage: python -m findex.bench_lab3 data/
"""
import argparse
import logging
from pathlib import Path

from findex.builder import build_index
from findex.models import DocMeta, Index, Posting
from findex.ranking import BM25, TfIdf, clear_query_cache, search


def run_sanity_checks(corpus_index: Index) -> None:
    print("=== 1. RANKING SANITY CHECKS ===")

    # Check 1: Rare term scores higher than frequent term (on real corpus)
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
    print("  1) Rare vs Frequent term:")
    print(f"     Rare  {rare_term!r} (df={len(rare_plist)}, tf={rare_doc.tf}) -> BM25={s_rare_bm25:.4f}, TF-IDF={s_rare_tfidf:.4f}")
    print(f"     Freq  {freq_term!r} (df={len(freq_plist)}, tf={freq_doc.tf}) -> BM25={s_freq_bm25:.4f}, TF-IDF={s_freq_tfidf:.4f}")
    assert s_rare_bm25 > s_freq_bm25 and s_rare_tfidf > s_freq_tfidf

    # Check 2: 20th repetition adds almost nothing (TF saturation)
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
    print(f"  2) TF Saturation (BM25, df={df_sample}, doc_len={dl}):")
    print(f"     tf=1 -> {bm25_1:.4f}, tf=2 -> {bm25_2:.4f} (gain +{gain_1_to_2:.4f})")
    print(f"     tf=19 -> {bm25_19:.4f}, tf=20 -> {bm25_20:.4f} (gain +{gain_19_to_20:.4f})")
    assert gain_19_to_20 < gain_1_to_2 * 0.05

    # Check 3: Short doc with 1 occurrence beats very long doc with 1 occurrence
    toy_index = Index(
        postings={"київ": [Posting(doc_id=0, tf=1), Posting(doc_id=1, tf=1)]},
        docs={
            0: DocMeta(doc_id=0, path="short.txt", title="Short doc", length=50),
            1: DocMeta(doc_id=1, path="long.txt", title="Very long doc", length=20000),
        },
    )
    res = search(toy_index, "київ", scorer=bm25, top_k=2, include_snippets=False)
    print("  3) Short vs Long doc (both tf=1):")
    print(f"     Short (len=50):  score={res[0].score:.4f} ({res[0].title})")
    print(f"     Long (len=20000): score={res[1].score:.4f} ({res[1].title})")
    assert res[0].doc_id == 0 and res[0].score > res[1].score


def run_precision_eval(index: Index) -> None:
    print()
    print("=== 2. PRECISION@5 EVALUATION (10 QUERIES: TF-IDF vs BM25) ===")
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

        def calc_p5(results, markers=relevant_markers) -> float:
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
        print(f"  {q:<55} | P@5 TF-IDF={p_tf:.2f} | BM25={p_bm:.2f}")

    mean_tfidf = sum(p5_tfidf_list) / len(p5_tfidf_list)
    mean_bm25 = sum(p5_bm25_list) / len(p5_bm25_list)
    print(f"  MEAN P@5: TF-IDF={mean_tfidf:.3f} | BM25={mean_bm25:.3f}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Run Lab 3 benchmarks")
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    clear_query_cache()
    index = build_index(args.root, with_positions=True)
    run_sanity_checks(index)
    run_precision_eval(index)

    print()
    print("=== 3. @timed & @lru_cache TRACE ===")
    demo_q = "київ AND (україна OR львів) NOT росія"
    _ = search(index, demo_q, scorer=BM25(), top_k=3, include_snippets=False)
    _ = search(index, demo_q, scorer=BM25(), top_k=3, include_snippets=False)


if __name__ == "__main__":
    main()
