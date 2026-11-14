"""Dispute lifecycle and verdict evaluation routes for Virtual Judge.

Core Functionality (Strictly per Project Proposal):
1. Two-sided dispute submission
2. 3-Tier AI Pipeline execution:
   - Layer 1: Logistic Regression text signals (Toxicity, Sarcasm, Sentiment) + Gatekeeper
   - Layer 2: Bayesian Network probabilistic support inference P(Support_A) vs P(Support_B)
   - Layer 3: Constraint Satisfaction Problem (CSP) output validation & inconclusive rules
3. Comprehensive Verdict display with explainable AI reasoning
4. System statistics and benchmark evaluation charts
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path when executed directly or imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datetime import datetime
from decimal import Decimal
from flask import Blueprint, request, jsonify, render_template, url_for
from sqlalchemy import func
from core.extensions import db
from core.models import Dispute, User, Submission, AIResult
from ai.ml_pipeline import run_full_pipeline

disputes_bp = Blueprint("disputes", __name__)


FALLACY_EXPLANATIONS = {
    "Ad Hominem": "Attacked the person, not the argument",
    "Strawman": "Twisted the opponent's words unfairly",
    "False Dichotomy": "Gave only two options when more exist",
    "Appeal to Emotion": "Used feelings instead of empirical evidence",
    "Hasty Generalization": "Drew an overly broad conclusion from limited evidence",
}


def _to_decimal(val, default="0.0") -> Decimal:
    """Safely converts input values to Decimal with fallback default."""
    if val is None:
        return Decimal(str(default))
    try:
        return Decimal(str(round(float(val), 4)))
    except Exception:
        return Decimal(str(default))


def _map_fallacies_for_display(fallacies):
    """Formats detected fallacy tags with human-readable explanations."""
    if not fallacies:
        return []
    output = []
    for item in fallacies:
        name = str(item)
        explanation = FALLACY_EXPLANATIONS.get(name)
        if explanation:
            output.append(f"{name}: {explanation}")
        else:
            output.append(name)
    return output


def _evaluate_dispute_pipeline(dispute: Dispute, sub_a: Submission, sub_b: Submission):
    """Executes the 3-tier hybrid AI pipeline and persists the resulting AIResult."""
    name_a = sub_a.user.username if sub_a.user else "Side A"
    name_b = sub_b.user.username if sub_b.user else "Side B"

    pipeline_result = run_full_pipeline(
        dispute_id=dispute.id,
        text_a=sub_a.argument_text,
        text_b=sub_b.argument_text,
        user_a_name=name_a,
        user_b_name=name_b,
        topic=dispute.title,
    )

    if pipeline_result.get("status") == "flagged":
        dispute.status = "flagged"
        db.session.commit()
        return {
            "message": "Dispute flagged by gatekeeper due to severe toxicity.",
            "redirect_url": url_for("disputes.get_dispute", dispute_id=dispute.id),
        }, 200

    if pipeline_result.get("status") == "error":
        return {
            "error": pipeline_result.get("message", "AI pipeline evaluation failed."),
        }, 500

    verdict = pipeline_result.get("verdict") or {}
    ml_scores = pipeline_result.get("ml_scores") or {}
    winner_name = verdict.get("winner")

    winner_user = None
    if sub_a.user and winner_name in [sub_a.user.username, name_a, "Side A"]:
        winner_user = sub_a.user
    elif sub_b.user and winner_name in [sub_b.user.username, name_b, "Side B"]:
        winner_user = sub_b.user
    elif winner_name in [name_a, "Side A"]:
        winner_user = sub_a.user or db.session.get(User, 1)
    elif winner_name in [name_b, "Side B"]:
        winner_user = sub_b.user or db.session.get(User, 2)

    winner_id = getattr(winner_user, "id", None)

    ai_res = AIResult(
        dispute_id=dispute.id,
        toxicity_score_a=_to_decimal(ml_scores.get("toxicity_a"), "0.0"),
        toxicity_score_b=_to_decimal(ml_scores.get("toxicity_b"), "0.0"),
        sentiment_a=str(ml_scores.get("sentiment_a") or "neutral"),
        sentiment_b=str(ml_scores.get("sentiment_b") or "neutral"),
        sarcasm_score_a=_to_decimal(ml_scores.get("sarcasm_a"), "0.0"),
        sarcasm_score_b=_to_decimal(ml_scores.get("sarcasm_b"), "0.0"),
        winner_user_id=winner_id,
        support_score_a=_to_decimal(verdict.get("support_score_a"), "0.5"),
        support_score_b=_to_decimal(verdict.get("support_score_b"), "0.5"),
        is_inconclusive=bool(verdict.get("is_inconclusive", False)),
        reasoning=str(verdict.get("reasoning") or "Evaluation completed."),
        confidence_score=_to_decimal(verdict.get("confidence_score"), "0.5"),
        fallacies_a=verdict.get("fallacies_a") or [],
        fallacies_b=verdict.get("fallacies_b") or [],
    )
    db.session.add(ai_res)
    dispute.status = "resolved"
    dispute.resolved_at = datetime.now()
    db.session.commit()

    return {
        "message": "Dispute evaluated successfully",
        "redirect_url": url_for("disputes.get_verdict", dispute_id=dispute.id),
    }, 200


# -------------------------------------------------------------------------
# Home Landing Page
# -------------------------------------------------------------------------
@disputes_bp.route("/")
def home():
    """Renders landing page with system overview, architecture, and recent cases."""
    stats = {
        "total_disputes": Dispute.query.count(),
        "total_verdicts": Dispute.query.filter_by(status="resolved").count(),
        "total_users": User.query.count(),
    }
    recent_disputes = (
        Dispute.query.order_by(Dispute.created_at.desc())
        .limit(6)
        .all()
    )
    return render_template(
        "index.html",
        stats=stats,
        recent_disputes=recent_disputes,
    )


# -------------------------------------------------------------------------
# Browse Disputes
# -------------------------------------------------------------------------
@disputes_bp.route("/disputes", methods=["GET"])
def list_disputes():
    """Lists disputes with optional search and status filtering."""
    query = Dispute.query

    status_filter = request.args.get("status", "").strip().lower()
    if status_filter in {"waiting", "active", "resolved", "flagged"}:
        query = query.filter_by(status=status_filter)

    search = request.args.get("search", "").strip()
    if search:
        query = query.filter(
            Dispute.title.ilike(f"%{search}%") | Dispute.description.ilike(f"%{search}%")
        )

    disputes = query.order_by(Dispute.created_at.desc()).all()
    return render_template(
        "dispute_view.html",
        disputes=disputes,
        search_term=search,
        status_filter=status_filter,
    )


# -------------------------------------------------------------------------
# Create Dispute
# -------------------------------------------------------------------------
@disputes_bp.route("/dispute/create", methods=["GET"])
def create_dispute_page():
    """Renders two-sided dispute submission page."""
    return render_template("create_dispute.html")


@disputes_bp.route("/dispute/create", methods=["POST"])
def create_dispute():
    """Creates a dispute and executes the 3-tier AI evaluation pipeline."""
    data = request.get_json(silent=True) or request.form
    title = (data.get("title") or "").strip()
    description = (data.get("description") or "").strip()

    text_a = (data.get("argument_text_a") or data.get("text_a") or "").strip()
    text_b = (data.get("argument_text_b") or data.get("text_b") or "").strip()

    if not title:
        return jsonify({"error": "Dispute topic/title is required"}), 400

    if not text_a:
        return jsonify({"error": "Argument text for Side A is required"}), 400

    user_a = db.session.get(User, 1) or User.query.filter_by(username="Side A").first()
    user_b = db.session.get(User, 2) or User.query.filter_by(username="Side B").first()
    user_a_id = user_a.id if user_a else 1
    user_b_id = user_b.id if user_b else 2

    dispute = Dispute(
        title=title,
        description=description,
        created_by=user_a_id,
        invited_user=user_b_id,
        status="waiting",
        is_public=True,
    )
    db.session.add(dispute)
    db.session.commit()

    # Save Side A Submission
    sub_a = Submission(dispute_id=dispute.id, user_id=user_a_id, argument_text=text_a)
    db.session.add(sub_a)
    db.session.commit()

    # If Side B argument is also provided, evaluate immediately
    if text_b:
        sub_b = Submission(dispute_id=dispute.id, user_id=user_b_id, argument_text=text_b)
        db.session.add(sub_b)
        db.session.commit()
        result_payload, status_code = _evaluate_dispute_pipeline(dispute, sub_a, sub_b)
        return jsonify(result_payload), status_code

    # Only Side A provided; waiting for Side B
    return jsonify({
        "message": "Dispute created. Waiting for opposing argument.",
        "redirect_url": url_for("disputes.get_dispute", dispute_id=dispute.id),
    }), 201


# -------------------------------------------------------------------------
# Single Dispute Details & Submit Opposing Argument
# -------------------------------------------------------------------------
@disputes_bp.route("/dispute/<int:dispute_id>", methods=["GET"])
def get_dispute(dispute_id):
    """Renders details of a single dispute and its submissions."""
    dispute = Dispute.query.get_or_404(dispute_id)
    submissions = Submission.query.filter_by(dispute_id=dispute.id).order_by(Submission.submitted_at.asc()).all()
    ai_result = AIResult.query.filter_by(dispute_id=dispute.id).first()

    return render_template(
        "dispute_view.html",
        dispute=dispute,
        submissions=submissions,
        ai_result=ai_result,
    )


@disputes_bp.route("/dispute/<int:dispute_id>/submit", methods=["POST"])
def submit_argument(dispute_id):
    """Submits the opposing argument and executes the 3-tier AI evaluation pipeline."""
    dispute = Dispute.query.get_or_404(dispute_id)
    data = request.get_json(silent=True) or request.form
    argument_text = (data.get("argument_text") or "").strip()

    if not argument_text:
        return jsonify({"error": "Argument text is required"}), 400

    submissions = Submission.query.filter_by(dispute_id=dispute.id).order_by(Submission.submitted_at.asc()).all()
    if len(submissions) >= 2 or dispute.status in ["resolved", "flagged"]:
        return jsonify({"error": "Both opposing arguments have already been submitted."}), 400

    user_b = db.session.get(User, 2) or User.query.filter_by(username="Side B").first()
    user_b_id = user_b.id if user_b else 2

    # Avoid duplicate user_id on unique constraint if submission 1 had user_b_id
    if submissions and submissions[0].user_id == user_b_id:
        user_b_id = 1

    submission = Submission(dispute_id=dispute.id, user_id=user_b_id, argument_text=argument_text)
    db.session.add(submission)
    db.session.commit()

    all_submissions = Submission.query.filter_by(dispute_id=dispute.id).order_by(Submission.submitted_at.asc()).all()
    if len(all_submissions) == 2:
        result_payload, status_code = _evaluate_dispute_pipeline(dispute, all_submissions[0], all_submissions[1])
        return jsonify(result_payload), status_code

    return jsonify({
        "message": "Argument submitted",
        "redirect_url": url_for("disputes.get_dispute", dispute_id=dispute.id),
    }), 200


# -------------------------------------------------------------------------
# Verdict View
# -------------------------------------------------------------------------
@disputes_bp.route("/verdict/<int:dispute_id>", methods=["GET"])
def get_verdict(dispute_id):
    """Renders the comprehensive multi-tier AI evaluation verdict."""
    dispute = Dispute.query.get_or_404(dispute_id)
    ai_result = AIResult.query.filter_by(dispute_id=dispute.id).first_or_404()
    submissions = Submission.query.filter_by(dispute_id=dispute.id).order_by(Submission.submitted_at.asc()).all()

    fallacies_a_display = _map_fallacies_for_display(ai_result.fallacies_a)
    fallacies_b_display = _map_fallacies_for_display(ai_result.fallacies_b)

    return render_template(
        "verdict.html",
        dispute=dispute,
        ai_result=ai_result,
        submissions=submissions,
        fallacies_a_display=fallacies_a_display,
        fallacies_b_display=fallacies_b_display,
    )


# -------------------------------------------------------------------------
# System Statistics & Evaluation Benchmarks
# -------------------------------------------------------------------------
@disputes_bp.route("/statistics", methods=["GET"])
def statistics_page():
    """Renders statistics and displays the 3 proposal evaluation charts."""
    total_disputes = Dispute.query.count()
    total_verdicts = Dispute.query.filter_by(status="resolved").count()
    inconclusive_count = AIResult.query.filter_by(is_inconclusive=True).count()

    avg_conf_row = db.session.query(func.avg(AIResult.confidence_score)).first()
    avg_confidence_pct = round(float(avg_conf_row[0] or 0.65) * 100, 1) if avg_conf_row else 65.0

    avg_tox_a = db.session.query(func.avg(AIResult.toxicity_score_a)).first()
    avg_tox_b = db.session.query(func.avg(AIResult.toxicity_score_b)).first()
    raw_tox_a = float(avg_tox_a[0] or 0.0) if avg_tox_a else 0.0
    raw_tox_b = float(avg_tox_b[0] or 0.0) if avg_tox_b else 0.0
    avg_toxicity_pct = round(((raw_tox_a + raw_tox_b) / 2.0) * 100, 1)

    return render_template(
        "statistics.html",
        total_disputes=total_disputes,
        total_verdicts=total_verdicts,
        inconclusive_count=inconclusive_count,
        avg_confidence_pct=avg_confidence_pct,
        avg_toxicity_pct=avg_toxicity_pct,
    )


if __name__ == "__main__":
    from core.app import create_app
    app = create_app()
    app.run(debug=True)
