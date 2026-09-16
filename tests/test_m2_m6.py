"""
Tests for M2-M6. Run with: pytest tests/ -v
No API keys required — all LLM calls are mocked.
"""

import pytest
from langchain.schema import Document

from src.retrieval.bm25_index import BM25Index, tokenize
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.retriever import RetrievedChunk
from src.generation.citations import CitedAnswer, Citation, _parse_response
from src.evaluation.evaluator import EvalResult, EvalSample, evaluate_simple


# ── M2: BM25 ─────────────────────────────────────────────────────────────────

DOCS = [
    Document(page_content="BERT is a transformer model pre-trained on masked language modeling.",
             metadata={"file_name": "dl.txt", "page": 1, "chunk_index": 0}),
    Document(page_content="Attention mechanisms allow the model to focus on relevant tokens.",
             metadata={"file_name": "dl.txt", "page": 2, "chunk_index": 1}),
    Document(page_content="Python is a high-level programming language.",
             metadata={"file_name": "prog.txt", "page": 1, "chunk_index": 2}),
]


def test_bm25_tokenizer():
    tokens = tokenize("Hello World 123")
    assert tokens == ["hello", "world", "123"]


def test_bm25_build_and_query():
    idx = BM25Index()
    idx.build(DOCS)
    results = idx.query("BERT transformer", top_k=2)
    assert len(results) == 2
    # BERT doc should be top result
    assert "BERT" in results[0][0].page_content


def test_bm25_normalised_scores():
    idx = BM25Index()
    idx.build(DOCS)
    results = idx.query("transformer", top_k=3)
    scores = [s for _, s in results]
    # Scores should be in [0, 1] and descending
    assert all(0 <= s <= 1.0 for s in scores)
    assert scores == sorted(scores, reverse=True)


# ── M2: RRF ──────────────────────────────────────────────────────────────────

def test_rrf_boosts_cross_list_agreement():
    doc_a = Document(page_content="Alpha", metadata={})
    doc_b = Document(page_content="Beta",  metadata={})
    doc_c = Document(page_content="Gamma", metadata={})

    # doc_a is #1 in list1, #2 in list2 → high RRF
    # doc_c is #3 in list1, #1 in list2 → medium RRF
    list1 = [(doc_a, 0.9), (doc_b, 0.7), (doc_c, 0.5)]
    list2 = [(doc_c, 0.8), (doc_a, 0.6), (doc_b, 0.4)]

    fused = reciprocal_rank_fusion([list1, list2], top_k=3)
    texts = [doc.page_content for doc, _ in fused]
    assert texts[0] == "Alpha"   # top in both lists


def test_rrf_scores_descending():
    doc_a = Document(page_content="A", metadata={})
    doc_b = Document(page_content="B", metadata={})
    list1 = [(doc_a, 1.0), (doc_b, 0.5)]
    list2 = [(doc_b, 1.0), (doc_a, 0.5)]
    fused = reciprocal_rank_fusion([list1, list2], top_k=2)
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)


# ── M5: Citations ─────────────────────────────────────────────────────────────

def make_chunks(n=3) -> list:
    return [
        RetrievedChunk(
            text=f"Chunk {i} content about topic {i}.",
            score=0.9 - i * 0.1,
            source="doc.pdf",
            page=i,
            chunk_index=i,
        )
        for i in range(n)
    ]


def test_parse_valid_citation_json():
    chunks = make_chunks(3)
    raw = '{"answer": "The answer is X.", "cited_chunks": [1, 3]}'
    result = _parse_response(raw, chunks)
    assert result.answer == "The answer is X."
    assert len(result.citations) == 2
    assert result.citations[0].page == 0   # chunk 1 → index 0
    assert result.citations[1].page == 2   # chunk 3 → index 2


def test_parse_invalid_json_fallback():
    chunks = make_chunks(2)
    result = _parse_response("not valid json at all", chunks)
    assert result.answer == "not valid json at all"
    assert result.citations == []


def test_cited_answer_out_of_range_indices():
    chunks = make_chunks(2)
    raw = '{"answer": "Answer.", "cited_chunks": [1, 99]}'  # 99 is out of range
    result = _parse_response(raw, chunks)
    assert len(result.citations) == 1   # only valid index kept


# ── M6: Evaluation ───────────────────────────────────────────────────────────

EVAL_SAMPLES = [
    EvalSample(
        question="What is BERT?",
        ground_truth="BERT is a transformer model for NLP.",
        answer="BERT stands for Bidirectional Encoder Representations from Transformers.",
        contexts=["BERT is a transformer-based model pre-trained on large text corpora."],
    ),
    EvalSample(
        question="What is attention?",
        ground_truth="Attention allows models to focus on relevant tokens.",
        answer="Attention mechanisms help the model focus on important parts of the input.",
        contexts=["Self-attention allows each token to attend to all other tokens."],
    ),
]


def test_evaluate_simple_returns_eval_result():
    result = evaluate_simple(EVAL_SAMPLES)
    assert isinstance(result, EvalResult)
    assert 0 <= result.faithfulness <= 1
    assert 0 <= result.answer_relevance <= 1
    assert 0 <= result.context_recall <= 1
    assert 0 <= result.context_precision <= 1


def test_ragas_score_is_harmonic_mean():
    r = EvalResult(faithfulness=0.8, answer_relevance=0.8,
                   context_recall=0.8, context_precision=0.8)
    assert r.ragas_score == pytest.approx(0.8, abs=0.01)


def test_ragas_score_zero_if_any_metric_zero():
    r = EvalResult(faithfulness=0.0, answer_relevance=0.9,
                   context_recall=0.9, context_precision=0.9)
    assert r.ragas_score == 0.0
