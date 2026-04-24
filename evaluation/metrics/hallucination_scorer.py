"""
SENTRY — Hallucination Scorer
================================
Quantifies hallucination in LLM responses by measuring how much
of the response can be traced back to the retrieved source chunks.

This implements the grounding accuracy metric described in the
proposal's evaluation framework (Section 3.4).

Two metrics are computed:

    1. Grounding Accuracy (0.0 – 1.0)
       Proportion of sentences in the response that contain at least
       one n-gram overlap with the retrieved context.
       Higher = more grounded = less hallucinated.

    2. Hallucination Rate (0.0 – 1.0)
       1 - grounding_accuracy.
       Proportion of sentences with no traceable overlap with context.
       Lower = better = fewer hallucinations.

Methodology note:
    This is an automated proxy metric — a complement to (not replacement
    for) expert human annotation. In your evaluation study, a subset of
    responses will be manually annotated by domain experts and compared
    against these automated scores for validation.

    The n-gram overlap approach is deliberately conservative:
    a sentence must share at least one meaningful n-gram with the
    retrieved context to be considered grounded. This avoids crediting
    responses for generic filler phrases.

Used by: evaluation/metrics/grounding_scorer.py
         tests/unit/test_hallucination_scorer.py
"""

import re
from typing import List, Dict, Any


class HallucinationScorer:
    """
    Automated hallucination scorer using n-gram overlap analysis.

    Parameters:
        ngram_size: Size of n-grams used for overlap comparison.
                    Default is 3 (trigrams) — small enough to catch
                    paraphrases, large enough to avoid false positives
                    from common stop-word sequences.
        min_sentence_words: Sentences shorter than this are skipped
                    (avoids scoring transitional phrases like "Note that").
    """

    def __init__(
        self,
        ngram_size: int = 3,
        min_sentence_words: int = 6,
    ) -> None:
        self.ngram_size = ngram_size
        self.min_sentence_words = min_sentence_words

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        response: str,
        context_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Score a response against its retrieved context chunks.

        Args:
            response:       The LLM-generated response text.
            context_chunks: The chunks retrieved and used to ground
                            the response (from pipeline result).

        Returns:
            Dict containing:
                - "grounding_accuracy":  Float 0–1. Proportion of
                                         sentences traceable to context.
                - "hallucination_rate":  Float 0–1. Inverse of above.
                - "total_sentences":     Number of sentences evaluated.
                - "grounded_sentences":  Sentences with context overlap.
                - "ungrounded_sentences":Sentences with no overlap.
                - "sentence_scores":     Per-sentence breakdown.
        """
        if not response or not response.strip():
            return self._empty_result()

        if not context_chunks:
            # No context = baseline mode — every sentence is ungrounded
            sentences = self._extract_sentences(response)
            return {
                "grounding_accuracy": 0.0,
                "hallucination_rate": 1.0,
                "total_sentences": len(sentences),
                "grounded_sentences": 0,
                "ungrounded_sentences": len(sentences),
                "sentence_scores": [
                    {"sentence": s, "grounded": False, "overlap_count": 0}
                    for s in sentences
                ],
            }

        # Build the full context corpus from all retrieved chunks
        context_text = " ".join(
            c.get("text", "") for c in context_chunks
        )
        context_ngrams = self._get_ngrams(context_text)

        sentences = self._extract_sentences(response)
        sentence_scores = []
        grounded_count = 0

        for sentence in sentences:
            overlap = self._count_overlap(sentence, context_ngrams)
            is_grounded = overlap > 0

            if is_grounded:
                grounded_count += 1

            sentence_scores.append({
                "sentence": sentence,
                "grounded": is_grounded,
                "overlap_count": overlap,
            })

        total = len(sentences)
        grounding_accuracy = (
            round(grounded_count / total, 4) if total > 0 else 0.0
        )

        return {
            "grounding_accuracy": grounding_accuracy,
            "hallucination_rate": round(1 - grounding_accuracy, 4),
            "total_sentences": total,
            "grounded_sentences": grounded_count,
            "ungrounded_sentences": total - grounded_count,
            "sentence_scores": sentence_scores,
        }

    def score_pair(
        self,
        grounded_result: Dict[str, Any],
        baseline_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Score both pipeline modes for the same query and return
        a comparative result. This is what the evaluation study
        calls for each participant interaction.

        Args:
            grounded_result: Full result dict from pipeline.query_grounded()
            baseline_result: Full result dict from pipeline.query_baseline()

        Returns:
            Comparative evaluation dict with scores for both modes
            and a delta showing improvement from grounding.
        """
        grounded_score = self.score(
            response=grounded_result["response"],
            context_chunks=grounded_result.get("retrieved_chunks", []),
        )
        baseline_score = self.score(
            response=baseline_result["response"],
            context_chunks=[],  # Baseline has no context
        )

        grounding_improvement = round(
            grounded_score["grounding_accuracy"]
            - baseline_score["grounding_accuracy"],
            4,
        )
        hallucination_reduction = round(
            baseline_score["hallucination_rate"]
            - grounded_score["hallucination_rate"],
            4,
        )

        return {
            "query": grounded_result["query"],
            "grounded": grounded_score,
            "baseline": baseline_score,
            "grounding_improvement": grounding_improvement,
            "hallucination_reduction": hallucination_reduction,
            "latency_cost_ms": round(
                grounded_result["total_ms"] - baseline_result["total_ms"], 2
            ),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences and filter out short ones.
        Handles numbered lists (e.g. "1. Do not click...")
        and bullet points as sentence boundaries.
        """
        # Normalise list markers to full stops for splitting
        text = re.sub(r"\n\s*\d+\.\s+", ". ", text)
        text = re.sub(r"\n\s*[-*]\s+", ". ", text)

        # Split on sentence-ending punctuation
        raw = re.split(r"(?<=[.!?])\s+", text.strip())

        sentences = []
        for s in raw:
            s = s.strip()
            if len(s.split()) >= self.min_sentence_words:
                sentences.append(s)

        return sentences

    def _get_ngrams(self, text: str) -> set:
        """
        Extract all n-grams from text as a set of tuples.
        Lowercased and stripped of punctuation for fuzzy matching.
        """
        words = self._tokenize(text)
        if len(words) < self.ngram_size:
            return set()
        return {
            tuple(words[i: i + self.ngram_size])
            for i in range(len(words) - self.ngram_size + 1)
        }

    def _count_overlap(self, sentence: str, context_ngrams: set) -> int:
        """
        Count how many n-grams from the sentence appear in context_ngrams.
        """
        sentence_ngrams = self._get_ngrams(sentence)
        return len(sentence_ngrams & context_ngrams)

    def _tokenize(self, text: str) -> List[str]:
        """Lowercase and remove punctuation, return word list."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return text.split()

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "grounding_accuracy": 0.0,
            "hallucination_rate": 1.0,
            "total_sentences": 0,
            "grounded_sentences": 0,
            "ungrounded_sentences": 0,
            "sentence_scores": [],
        }