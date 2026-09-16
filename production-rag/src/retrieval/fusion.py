"""
Reciprocal Rank Fusion (RRF) — M2: merging dense + sparse rankings.

THE PROBLEM:
  Dense retriever returns: [chunk_A=0.91, chunk_B=0.87, chunk_C=0.72, ...]
  BM25 retriever returns:  [chunk_C=0.95, chunk_A=0.60, chunk_D=0.55, ...]

  How do we combine them? We can't average raw scores — they live on different
  scales (cosine similarity vs BM25 Okapi scores).

RRF SOLUTION (Cormack et al. 2009):
  Forget the raw scores. Only use the RANK position.
  RRF_score(chunk) = Σ  1 / (k + rank_in_list_i)
                    i ∈ retrievers

  Where k=60 is a constant that dampens the impact of very high-ranked items.

EXAMPLE:
  chunk_A: rank 1 in dense, rank 2 in BM25
    RRF = 1/(60+1) + 1/(60+2) = 0.01639 + 0.01613 = 0.03252

  chunk_C: rank 3 in dense, rank 1 in BM25
    RRF = 1/(60+3) + 1/(60+1) = 0.01587 + 0.01639 = 0.03226

  chunk_A wins — it was top-ranked by the more precise dense retriever.
  But chunk_C is close — BM25 strongly endorsed it.

WHY k=60?
  Empirically found to work well. Higher k → less winner-take-all (ranks matter
  more equally). Lower k → top rank dominates aggressively.
"""

from typing import Dict, List, Tuple

from langchain_core.documents import Document


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[Document, float]]],
    k: int = 60,
    top_k: int = 10,
) -> List[Tuple[Document, float]]:
    """
    Fuse multiple ranked lists into a single ranking using RRF.

    Args:
        ranked_lists: Each element is a list of (Document, score) sorted best-first.
                      Scores are ignored — only rank position matters.
        k:            RRF constant (default 60, per original paper).
        top_k:        How many fused results to return.

    Returns:
        List of (Document, rrf_score) sorted best-first.
        rrf_score is in (0, 1] — higher is better.
    """
    # Use page_content as dedup key (chunk text is unique per chunk)
    rrf_scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    for ranked_list in ranked_lists:
        for rank, (doc, _score) in enumerate(ranked_list, start=1):
            key = doc.page_content  # unique identifier

            if key not in doc_map:
                doc_map[key] = doc
                rrf_scores[key] = 0.0

            rrf_scores[key] += 1.0 / (k + rank)

    # Sort by RRF score descending
    sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)

    return [
        (doc_map[key], round(rrf_scores[key], 6))
        for key in sorted_keys[:top_k]
    ]
