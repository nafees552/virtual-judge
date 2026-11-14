"""SQLAlchemy models representing Virtual Judge database entities and relationships.

Core Domain:
- User: Disputant participant identity (Side A / Side B)
- Dispute: Paired debate topic and status
- Submission: Argument text submitted for each side
- AIResult: Multi-tier AI evaluation results (LR signals, Bayesian probabilities, CSP verdict)
"""

from datetime import datetime
from flask_login import UserMixin
from sqlalchemy import UniqueConstraint
from core.extensions import db, login_manager


class User(UserMixin, db.Model):
    """Application users / disputant participants."""

    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False, default="user@virtualjudge.local")
    password_hash = db.Column(db.String(255), nullable=True, default="")
    role = db.Column(db.String(20), nullable=False, default="disputant")
    reputation_score = db.Column(db.Integer, nullable=False, default=0)
    is_active = db.Column(db.Boolean, nullable=False, default=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    created_disputes = db.relationship("Dispute", foreign_keys="Dispute.created_by", back_populates="creator")
    invited_disputes = db.relationship("Dispute", foreign_keys="Dispute.invited_user", back_populates="invited")
    submissions = db.relationship("Submission", back_populates="user", cascade="all, delete-orphan")
    won_results = db.relationship("AIResult", back_populates="winner")

    def __init__(
        self,
        id=None,
        username="",
        email="",
        password_hash="",
        role="disputant",
        reputation_score=0,
        is_active=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if id is not None:
            self.id = id
        self.username = username
        self.email = email
        self.password_hash = password_hash
        self.role = role
        self.reputation_score = reputation_score
        self.is_active = is_active


@login_manager.user_loader
def load_user(user_id):
    """Loads a user by id for Flask-Login session compatibility."""
    return db.session.get(User, int(user_id))


class Dispute(db.Model):
    """A debate/dispute containing two opposing arguments."""

    __tablename__ = "disputes"

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=True)
    created_by = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    invited_user = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    status = db.Column(db.String(20), nullable=False, default="waiting")  # waiting, active, resolved, flagged
    is_public = db.Column(db.Boolean, nullable=False, default=True)
    moderation_mode = db.Column(db.String(20), nullable=False, default="public")
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    resolved_at = db.Column(db.DateTime, nullable=True)
    moderator_note = db.Column(db.Text, nullable=True)
    review_state = db.Column(db.String(32), nullable=True)

    creator = db.relationship("User", foreign_keys=[created_by], back_populates="created_disputes")
    invited = db.relationship("User", foreign_keys=[invited_user], back_populates="invited_disputes")
    submissions = db.relationship("Submission", back_populates="dispute", cascade="all, delete-orphan")
    ai_result = db.relationship("AIResult", back_populates="dispute", uselist=False, cascade="all, delete-orphan")

    def __init__(
        self,
        title="",
        description=None,
        created_by=None,
        invited_user=None,
        status="waiting",
        is_public=True,
        moderation_mode="public",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.title = title
        self.description = description
        self.created_by = created_by
        self.invited_user = invited_user
        self.status = status
        self.is_public = is_public
        self.moderation_mode = moderation_mode


class Submission(db.Model):
    """Argument text submitted for a specific dispute."""

    __tablename__ = "submissions"
    __table_args__ = (UniqueConstraint("dispute_id", "user_id", name="uq_dispute_user_submission"),)

    id = db.Column(db.Integer, primary_key=True)
    dispute_id = db.Column(db.Integer, db.ForeignKey("disputes.id", ondelete="CASCADE"), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    argument_text = db.Column(db.Text, nullable=False)
    submitted_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    dispute = db.relationship("Dispute", back_populates="submissions")
    user = db.relationship("User", back_populates="submissions")

    def __init__(
        self,
        dispute_id=None,
        user_id=None,
        argument_text="",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dispute_id = dispute_id
        self.user_id = user_id
        self.argument_text = argument_text


class AIResult(db.Model):
    """AI analysis result linked to an evaluated dispute."""

    __tablename__ = "airesults"

    id = db.Column(db.Integer, primary_key=True)
    dispute_id = db.Column(db.Integer, db.ForeignKey("disputes.id", ondelete="CASCADE"), nullable=False, unique=True)
    toxicity_score_a = db.Column(db.Numeric(5, 4), nullable=False)
    toxicity_score_b = db.Column(db.Numeric(5, 4), nullable=False)
    sentiment_a = db.Column(db.String(20), nullable=False)
    sentiment_b = db.Column(db.String(20), nullable=False)
    sarcasm_score_a = db.Column(db.Numeric(5, 4), nullable=False)
    sarcasm_score_b = db.Column(db.Numeric(5, 4), nullable=False)
    winner_user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=True)
    support_score_a = db.Column(db.Numeric(5, 4), nullable=True)
    support_score_b = db.Column(db.Numeric(5, 4), nullable=True)
    is_inconclusive = db.Column(db.Boolean, nullable=False, default=False)
    reasoning = db.Column(db.Text, nullable=False)
    confidence_score = db.Column(db.Numeric(5, 4), nullable=False)
    fallacies_a = db.Column(db.JSON, nullable=True)
    fallacies_b = db.Column(db.JSON, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    dispute = db.relationship("Dispute", back_populates="ai_result")
    winner = db.relationship("User", back_populates="won_results")

    def __init__(
        self,
        dispute_id=None,
        toxicity_score_a=0.0,
        toxicity_score_b=0.0,
        sentiment_a="neutral",
        sentiment_b="neutral",
        sarcasm_score_a=0.0,
        sarcasm_score_b=0.0,
        winner_user_id=None,
        support_score_a=0.5,
        support_score_b=0.5,
        is_inconclusive=False,
        reasoning="",
        confidence_score=0.5,
        fallacies_a=None,
        fallacies_b=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dispute_id = dispute_id
        self.toxicity_score_a = toxicity_score_a
        self.toxicity_score_b = toxicity_score_b
        self.sentiment_a = sentiment_a
        self.sentiment_b = sentiment_b
        self.sarcasm_score_a = sarcasm_score_a
        self.sarcasm_score_b = sarcasm_score_b
        self.winner_user_id = winner_user_id
        self.support_score_a = support_score_a
        self.support_score_b = support_score_b
        self.is_inconclusive = is_inconclusive
        self.reasoning = reasoning
        self.confidence_score = confidence_score
        self.fallacies_a = fallacies_a if fallacies_a is not None else []
        self.fallacies_b = fallacies_b if fallacies_b is not None else []
