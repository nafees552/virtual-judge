"""Rule-based Weighted Scoring Baseline.

FAST — National University of Computer and Emerging Sciences
Course: AL-2002 Artificial Intelligence (Laboratory) · Fall 2026
Component: Experimental Baseline for Virtual Judge

Role:
A simple heuristic baseline that calculates a weighted score for each argument
using raw keyword counts and surface features without Bayesian probabilistic
reasoning or CSP constraint validation. Used to benchmark the multi-tier
hybrid LR + Bayesian + CSP pipeline.
"""

import re
import time


EVIDENCE_KEYWORDS = {
    "study", "studies", "research", "experiment", "survey", "trial",
    "dataset", "empirical", "data", "statistics", "percent", "percentage",
    "source", "citation", "reference", "report", "published", "journal",
    "proven", "analysis", "evidence", "facts", "factual",
}

POSITIVE_WORDS = {
    "benefit", "advantage", "positive", "improve", "superior", "efficient",
    "effective", "solution", "support", "progress", "success", "valuable",
    "constructive", "sound", "valid", "productive",
}

NEGATIVE_WORDS = {
    "harmful", "detrimental", "negative", "damage", "flaw", "inefficient",
    "problem", "failure", "risk", "hazard", "threat", "weakness", "bad",
    "worse", "disadvantage", "defect",
}

TOXIC_KEYWORDS = {
    "idiot", "stupid", "dumb", "moron", "shut up", "loser", "hate",
    "trash", "ugly", "scum", "bitch", "corrupt", "clueless",
}

SARCASM_CUES = {
    "brilliant", "genius", "obviously", "as if", "surely", "clearly",
    "wow", "oh really", "whatever", "big surprise",
}


class RuleBasedBaseline:
    """Fixed-weight heuristic scoring baseline without Bayesian inference or CSP."""

    def __init__(self,
                 w_length: float = 0.15,
                 w_evidence: float = 0.40,
                 w_sentiment: float = 0.15,
                 w_toxic: float = 0.50,
                 w_sarcasm: float = 0.20):
        self.w_length = w_length
        self.w_evidence = w_evidence
        self.w_sentiment = w_sentiment
        self.w_toxic = w_toxic
        self.w_sarcasm = w_sarcasm

    def score_text(self, text: str) -> dict:
        """Calculates fixed-weight score from word counts and dictionary matching."""
        lower = (text or "").lower()
        words = re.findall(r"\b\w+\b", lower)
        word_count = len(words)

        ev_count = sum(1 for w in words if w in EVIDENCE_KEYWORDS)
        pos_count = sum(1 for w in words if w in POSITIVE_WORDS)
        neg_count = sum(1 for w in words if w in NEGATIVE_WORDS)
        toxic_count = sum(1 for w in words if w in TOXIC_KEYWORDS)
        sarcasm_count = sum(1 for w in words if w in SARCASM_CUES)

        # Length score normalized up to 100 words
        len_score = min(1.0, word_count / 100.0)
        ev_score = min(1.0, ev_count / 4.0)
        sent_balance = (pos_count - neg_count) / max(1, pos_count + neg_count)

        total_score = (
            (self.w_length * len_score)
            + (self.w_evidence * ev_score)
            + (self.w_sentiment * max(0.0, sent_balance))
            - (self.w_toxic * min(1.0, toxic_count / 2.0))
            - (self.w_sarcasm * min(1.0, sarcasm_count / 2.0))
        )

        return {
            "score": round(max(0.0, total_score), 4),
            "word_count": word_count,
            "evidence_count": ev_count,
            "toxic_count": toxic_count,
            "sarcasm_count": sarcasm_count,
        }

    def evaluate_dispute(self, topic: str, text_a: str, text_b: str, user_a_name: str = "A", user_b_name: str = "B") -> dict:
        """Evaluates dispute based purely on which argument achieves the higher weighted score."""
        start_time = time.time()
        score_a = self.score_text(text_a)
        score_b = self.score_text(text_b)
        elapsed = time.time() - start_time

        s_a = score_a["score"]
        s_b = score_b["score"]

        # Notice: The baseline has NO CSP and NO tolerance margin; whoever has slightly higher score wins
        if abs(s_a - s_b) < 0.001:
            winner = "Tie"
            decision = "Tie"
        elif s_a > s_b:
            winner = "A"
            decision = "Side A"
        else:
            winner = "B"
            decision = "Side B"

        return {
            "decision": decision,
            "winner": winner,
            "score_a": s_a,
            "score_b": s_b,
            "latency": elapsed,
            "metrics_a": score_a,
            "metrics_b": score_b,
        }

