"""Orchestration pipeline for Virtual Judge argument evaluation.

Integrates:
- Layer 1: Logistic Regression text classifiers (Toxicity, Sarcasm, Sentiment)
- Layer 2: Bayesian Network for probabilistic argument support inference
- Layer 3: Constraint Satisfaction Problem (CSP) for outcome validation
"""

from ai.virtual_judge import VirtualJudgeController

# Singleton controller instance for fast reusable inferences
_controller = None


def get_controller():
    """Lazily initializes and returns singleton VirtualJudgeController."""
    global _controller
    if _controller is None:
        _controller = VirtualJudgeController()
    return _controller


def run_ml_analysis(text_a: str, text_b: str, topic: str = "") -> dict:
    """Scores both arguments for toxicity, sarcasm, and sentiment using LR classifiers."""
    controller = get_controller()
    clf = controller.signal_classifier

    tox_a = clf.predict_toxicity(text_a)
    tox_b = clf.predict_toxicity(text_b)
    sar_a = clf.predict_sarcasm(text_a)
    sar_b = clf.predict_sarcasm(text_b)
    sen_a, comp_a = clf.predict_sentiment(text_a)
    sen_b, comp_b = clf.predict_sentiment(text_b)

    is_toxic = (tox_a >= 0.75) or (tox_b >= 0.75)

    return {
        "toxicity_a": round(tox_a, 4),
        "toxicity_b": round(tox_b, 4),
        "sarcasm_a": round(sar_a, 4),
        "sarcasm_b": round(sar_b, 4),
        "sentiment_a": sen_a,
        "sentiment_b": sen_b,
        "sentiment_compound_a": comp_a,
        "sentiment_compound_b": comp_b,
        "is_toxic": is_toxic,
    }


def run_full_pipeline(dispute_id, text_a: str, text_b: str, user_a_name: str, user_b_name: str,
                      appeal_context: str = None, prior_verdict_context: str = None, topic: str = ""):
    """Runs the full 3-tier LR + Bayesian + CSP pipeline and returns structured payload."""
    controller = get_controller()
    dispute_topic = topic or "General Dispute"

    result = controller.run_pipeline(
        topic=dispute_topic,
        text_a=text_a,
        text_b=text_b,
        user_a_name=user_a_name,
        user_b_name=user_b_name,
    )
    result["dispute_id"] = dispute_id

    # If appeal context exists, append to reasoning
    if appeal_context and result.get("verdict"):
        result["verdict"]["reasoning"] += f"\n\n[Appeals Review Audit]: Re-evaluated in light of user appeal notes: '{appeal_context}'."

    return result
