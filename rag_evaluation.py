
import math
from statistics import mean
from dataclasses import dataclass

@dataclass
class EvaluationMetrics:
    retrieved_chunks: int
    reranked_chunks: int
    average_reranker_score: float
    confidence: float
    retrieval_time: float
    rerank_time: float
    generation_time: float
    @property
    def total_time(self):
        return (
        self.retrieval_time
        + self.rerank_time
        + self.generation_time
    )
    def format_report(self):
        return f"""
==============================
RAG Evaluation Report
==============================
Retrieved Chunks       : {self.retrieved_chunks}
Chunks After Rerank    : {self.reranked_chunks}
Average Reranker Score : {self.average_reranker_score:.2f}
Confidence             : {self.confidence:.1f}%
Retrieval Time         : {self.retrieval_time:.3f} s
Rerank Time            : {self.rerank_time:.3f} s
Generation Time        : {self.generation_time:.3f} s
Total Time             : {self.total_time:.3f} s
"""
    
def normalize_score(scores):
    if not scores:
        return []

    mn = min(scores)
    mx = max(scores)

    if mx == mn:
        return [10.0] * len(scores)

    return [
        ((s - mn) / (mx - mn)) * 10
        for s in scores
    ]


def evaluate_pipeline(
    retrieved_count,
    reranked_scores,
    retrieval_time,
    rerank_time,
    generation_time,
):

    normalized_scores = normalize_score(reranked_scores)

    avg_normalized = (
    mean(normalized_scores)
    if normalized_scores
    else 0
)

    confidence = avg_normalized * 10

    return EvaluationMetrics(
        retrieved_chunks=retrieved_count,
        reranked_chunks=len(reranked_scores),
        average_reranker_score=avg_normalized,
        confidence=confidence,
        retrieval_time=retrieval_time,
        rerank_time=rerank_time,
        generation_time=generation_time,
)