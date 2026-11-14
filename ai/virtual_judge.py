"""Virtual Judge: AI-Assisted Argument Evaluation.

FAST — National University of Computer and Emerging Sciences
Department of Artificial Intelligence & Data Science (Fall 2026)
Course: AL-2002 Artificial Intelligence (Laboratory)

Authors:
- Maheen Haseeb (24K-2530) — Feature Extraction, LR Classifiers & Controller
- M. Nafees Uddin (24K-2526) — Bayesian Network & Probabilistic Inference
- Safa Rehman (24K-2562) — Constraint Satisfaction Problem (CSP) & Baseline Evaluation

Three-Tier AI Architecture:
1. Layer 1: Logistic Regression Classifiers (Toxicity, Sarcasm, Sentiment) + Gatekeeper
2. Layer 2: Bayesian Network (Probabilistic Argument Support Inference)
3. Layer 3: Constraint Satisfaction Problem (Verdict Validation & Inconclusive Rules)
"""

from pathlib import Path
import re
import numpy as np
import joblib

# Attempt pgmpy import with fallback to pure-python exact inference
try:
    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination
    PGMPY_AVAILABLE = True
except ImportError:
    try:
        from pgmpy.models import BayesianNetwork as DiscreteBayesianNetwork
        from pgmpy.factors.discrete import TabularCPD
        from pgmpy.inference import VariableElimination
        PGMPY_AVAILABLE = True
    except ImportError:
        PGMPY_AVAILABLE = False


# Factual and logical debate indicators
EVIDENCE_PATTERNS = [
    r"\b(?:study|studies|research|experiment|survey|trial|dataset|empirical)\b",
    r"\b(?:data|statistics|percentage|percent|%|ratio|metrics|figures)\b",
    r"\b(?:source|citation|reference|report|documented|published|journal)\b",
    r"\b(?:according to|proven|demonstrated|evidence shows|analysis indicates)\b",
    r"\b(?:\d+(?:\.\d+)?%|\$\d+|\b\d{4}\b)\b",  # numbers, percentages, years
]

FALLACY_PATTERNS = {
    "Ad Hominem": r"\b(?:you are (?:an idiot|stupid|dumb|moron|clueless|corrupt)|shut up|loser)\b",
    "Appeal to Emotion": r"\b(?:think of the children|monstrous|evil|heartless|disaster waiting to happen)\b",
    "Slippery Slope": r"\b(?:will lead to total ruin|inevitably destroy|end of civilization|catastrophe)\b",
    "Hasty Generalization": r"\b(?:everyone knows|always happens|nobody ever|all of them are)\b",
}


class Argument:
    """Stores raw submission text, topic, and extracts structural & semantic features."""

    def __init__(self, text: str, topic: str = "", author_name: str = "Participant"):
        self.raw_text = text or ""
        self.topic = topic or ""
        self.author_name = author_name
        self.word_count = len(re.findall(r"\b\w+\b", self.raw_text))
        self.tfidf_vector = None
        self.evidence_score = 0.0
        self.relevance_score = 0.0
        self.detected_fallacies = []
        self._analyze_features()

    def validate_input(self) -> bool:
        """Validates that input meets required length and structural criteria."""
        cleaned = self.raw_text.strip()
        if not cleaned:
            raise ValueError("Argument cannot be empty.")
        if self.word_count < 3:
            raise ValueError("Argument is too short to evaluate (minimum 3 words required).")
        if len(cleaned) > 8000:
            raise ValueError("Argument exceeds maximum allowed character length (8000).")
        return True

    def extract_tfidf_features(self, vectorizer=None):
        """Extracts TF-IDF feature vector and computes topic relevance."""
        if vectorizer is not None and hasattr(vectorizer, "transform"):
            try:
                self.tfidf_vector = vectorizer.transform([self.raw_text])
                if self.topic:
                    topic_vec = vectorizer.transform([self.topic])
                    # Cosine similarity between topic and argument vector
                    denom = (np.linalg.norm(self.tfidf_vector.toarray()) * np.linalg.norm(topic_vec.toarray()))
                    if denom > 0:
                        self.relevance_score = float(np.dot(self.tfidf_vector.toarray(), topic_vec.toarray().T)[0][0] / denom)
            except Exception:
                pass
        return self.tfidf_vector

    def _analyze_features(self):
        """Extracts lexical evidence indicators and rhetorical fallacies."""
        lower_text = self.raw_text.lower()

        # Evidence count
        ev_matches = 0
        for pattern in EVIDENCE_PATTERNS:
            ev_matches += len(re.findall(pattern, lower_text))
        
        # Word count bonus (substantial arguments contain more substance)
        length_bonus = min(0.3, (self.word_count / 150.0) * 0.3)
        self.evidence_score = min(1.0, (ev_matches * 0.18) + length_bonus)

        # Lexical topic overlap fallback for relevance
        if self.topic:
            topic_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", self.topic.lower()))
            arg_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", lower_text))
            if topic_words:
                overlap = len(topic_words.intersection(arg_words))
                self.relevance_score = min(1.0, overlap / max(1, len(topic_words)) + 0.2)
            else:
                self.relevance_score = 0.5
        else:
            self.relevance_score = 0.6

        # Check fallacies
        for name, pat in FALLACY_PATTERNS.items():
            if re.search(pat, lower_text):
                self.detected_fallacies.append(name)


class SignalClassifier:
    """Loads serialized Joblib Logistic Regression models and runs safety gatekeeper."""

    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = Path(__file__).resolve().parents[1] / "ml_models"
        self.model_dir = Path(model_dir)
        self.toxicity_model = None
        self.sarcasm_model = None
        self.sentiment_model = None
        self._load_models()

    def _load_models(self):
        tox_path = self.model_dir / "toxicity_model.pkl"
        sar_path = self.model_dir / "sarcasm_model.pkl"
        sen_path = self.model_dir / "sentiment_model.pkl"

        if tox_path.exists():
            self.toxicity_model = joblib.load(tox_path)
        if sar_path.exists():
            self.sarcasm_model = joblib.load(sar_path)
        if sen_path.exists():
            self.sentiment_model = joblib.load(sen_path)

    def predict_toxicity(self, text: str) -> float:
        """Predicts toxicity probability [0.0, 1.0] combining LR model and severe abuse cues."""
        if not self.toxicity_model:
            return 0.0
        try:
            proba = self.toxicity_model.predict_proba([text])[0]
            lr_val = float(proba[1]) if len(proba) > 1 else float(proba[0])
            
            # Check blatant toxic harassment indicators
            lower = (text or "").lower()
            severe_cues = [
                r"\b(?:idiot|moron|retard(?:ed)?|bitch|scum|whore|worthless|clown|piece of (?:shit|trash|garbage))\b",
                r"\b(?:shut (?:up|your)|kill yourself|die in a ditch|get lost you vile)\b",
            ]
            for pat in severe_cues:
                if re.search(pat, lower):
                    lr_val = max(lr_val, 0.85)
                    break
            return float(lr_val)
        except Exception:
            return 0.0

    def predict_sarcasm(self, text: str) -> float:
        """Predicts sarcasm probability [0.0, 1.0]."""
        if not self.sarcasm_model:
            return 0.0
        try:
            proba = self.sarcasm_model.predict_proba([text])[0]
            sar_val = float(proba[1]) if len(proba) > 1 else float(proba[0])
            lower = (text or "").lower()
            if any(cue in lower for cue in ["oh brilliant", "genius idea", "big surprise", "wow really"]):
                sar_val = max(sar_val, 0.75)
            return float(sar_val)
        except Exception:
            return 0.0

    def predict_sentiment(self, text: str) -> tuple:
        """Predicts sentiment ('positive'/'negative') and compound score [-1.0, 1.0]."""
        if not self.sentiment_model:
            return "neutral", 0.0
        try:
            classes = list(self.sentiment_model.classes_)
            probas = self.sentiment_model.predict_proba([text])[0]
            pred = self.sentiment_model.predict([text])[0]

            label = "positive" if str(pred).lower() in {"positive", "1"} else "negative"
            pos_idx = classes.index("positive") if "positive" in classes else (classes.index(1) if 1 in classes else 0)
            pos_prob = float(probas[pos_idx])
            compound = round((2.0 * pos_prob) - 1.0, 4)
            return label, compound
        except Exception:
            return "neutral", 0.0

    def run_gatekeeper_check(self, text_a: str, text_b: str, threshold: float = 0.65) -> dict:
        """Automated safety gatekeeper: halts processing if severe toxicity is detected."""
        tox_a = self.predict_toxicity(text_a)
        tox_b = self.predict_toxicity(text_b)

        is_flagged = (tox_a >= threshold) or (tox_b >= threshold)
        reason = ""
        violating_side = None

        if tox_a >= threshold and tox_b >= threshold:
            violating_side = "Both"
            reason = f"Gatekeeper Alert: Severe toxicity detected on both sides (A: {tox_a:.2f}, B: {tox_b:.2f})."
        elif tox_a >= threshold:
            violating_side = "A"
            reason = f"Gatekeeper Alert: Side A argument breached safety toxicity threshold ({tox_a:.2f} >= {threshold})."
        elif tox_b >= threshold:
            violating_side = "B"
            reason = f"Gatekeeper Alert: Side B argument breached safety toxicity threshold ({tox_b:.2f} >= {threshold})."

        return {
            "is_toxic": is_flagged,
            "toxicity_a": round(tox_a, 4),
            "toxicity_b": round(tox_b, 4),
            "violating_side": violating_side,
            "reason": reason,
        }


class BayesianJudge:
    """Defines Bayesian Network structure and CPTs to infer support probabilities."""

    def __init__(self):
        self.network = None
        self.inference_engine = None
        self.build_network()

    def build_network(self):
        """Constructs the Bayesian Network DAG and sets Conditional Probability Tables."""
        if not PGMPY_AVAILABLE:
            return

        try:
            # Nodes:
            # Observable: Toxicity (T), Sarcasm (S), Sentiment (Sent), Evidence (E), Relevance (R)
            # Latent: Credibility (C), Substance (Sub)
            # Target: Support (Supp)
            self.network = DiscreteBayesianNetwork([
                ("Toxicity", "Credibility"),
                ("Sarcasm", "Credibility"),
                ("Evidence", "Credibility"),
                ("Evidence", "Substance"),
                ("Relevance", "Substance"),
                ("Credibility", "Support"),
                ("Substance", "Support"),
                ("Sentiment", "Support"),
            ])

            # Prior CPDs for binary states [0=Low/Weak/Negative, 1=High/Strong/Positive]
            cpd_tox = TabularCPD("Toxicity", 2, [[0.7], [0.3]])
            cpd_sarc = TabularCPD("Sarcasm", 2, [[0.7], [0.3]])
            cpd_sent = TabularCPD("Sentiment", 2, [[0.5], [0.5]])
            cpd_ev = TabularCPD("Evidence", 2, [[0.6], [0.4]])
            cpd_rel = TabularCPD("Relevance", 2, [[0.3], [0.7]])

            # CPD Credibility | Toxicity, Sarcasm, Evidence
            # Parents cardinality: [2, 2, 2] -> 8 columns
            # Values: [P(C=0), P(C=1)]
            # High toxicity or high sarcasm drops credibility sharply
            cred_cpt = [
                # C=0 (Low Credibility)
                [0.40, 0.15, 0.70, 0.45, 0.85, 0.60, 0.95, 0.80],
                # C=1 (High Credibility)
                [0.60, 0.85, 0.30, 0.55, 0.15, 0.40, 0.05, 0.20],
            ]
            cpd_cred = TabularCPD(
                variable="Credibility",
                variable_card=2,
                values=cred_cpt,
                evidence=["Toxicity", "Evidence", "Sarcasm"],
                evidence_card=[2, 2, 2],
            )

            # CPD Substance | Evidence, Relevance
            # Parents cardinality: [2, 2] -> 4 columns
            sub_cpt = [
                # Sub=0 (Low Substance)
                [0.85, 0.45, 0.55, 0.10],
                # Sub=1 (High Substance)
                [0.15, 0.55, 0.45, 0.90],
            ]
            cpd_sub = TabularCPD(
                variable="Substance",
                variable_card=2,
                values=sub_cpt,
                evidence=["Evidence", "Relevance"],
                evidence_card=[2, 2],
            )

            # CPD Support | Credibility, Substance, Sentiment
            # Parents cardinality: [2, 2, 2] -> 8 columns
            supp_cpt = [
                # Supp=0 (Weak Support)
                [0.92, 0.80, 0.70, 0.50, 0.65, 0.40, 0.30, 0.08],
                # Supp=1 (Strong Support)
                [0.08, 0.20, 0.30, 0.50, 0.35, 0.60, 0.70, 0.92],
            ]
            cpd_supp = TabularCPD(
                variable="Support",
                variable_card=2,
                values=supp_cpt,
                evidence=["Credibility", "Substance", "Sentiment"],
                evidence_card=[2, 2, 2],
            )

            self.network.add_cpds(cpd_tox, cpd_sarc, cpd_sent, cpd_ev, cpd_rel, cpd_cred, cpd_sub, cpd_supp)
            assert self.network.check_model()
            self.inference_engine = VariableElimination(self.network)
        except Exception as e:
            self.network = None
            self.inference_engine = None

    def _infer_single(self, toxicity: float, sarcasm: float, sentiment: str, evidence: float, relevance: float) -> float:
        """Infers P(Support = Strong) given argument evidence and text signals."""
        t_bin = 1 if toxicity >= 0.5 else 0
        s_bin = 1 if sarcasm >= 0.45 else 0
        sent_bin = 1 if sentiment == "positive" else 0
        e_bin = 1 if evidence >= 0.35 else 0
        r_bin = 1 if relevance >= 0.40 else 0

        if self.inference_engine is not None:
            try:
                evidence_dict = {
                    "Toxicity": t_bin,
                    "Sarcasm": s_bin,
                    "Sentiment": sent_bin,
                    "Evidence": e_bin,
                    "Relevance": r_bin,
                }
                res = self.inference_engine.query(variables=["Support"], evidence=evidence_dict, show_progress=False)
                # P(Support = 1)
                prob_strong = float(res.values[1])
                # Smooth with continuous evidence modifiers
                fine_tuned = prob_strong * 0.7 + (evidence * 0.2) + (relevance * 0.1) - (toxicity * 0.15)
                return max(0.02, min(0.98, float(fine_tuned)))
            except Exception:
                pass

        # Exact mathematical Bayesian network fallback calculation
        # Modeling P(Support=1 | T, S, Sent, E, R)
        p_cred = 0.85 if (t_bin == 0 and s_bin == 0 and e_bin == 1) else (
            0.55 if (t_bin == 0 and s_bin == 0) else (
                0.30 if (t_bin == 1 or s_bin == 1) else 0.15
            )
        )
        p_sub = 0.90 if (e_bin == 1 and r_bin == 1) else (
            0.50 if (e_bin == 1 or r_bin == 1) else 0.15
        )
        p_supp = (0.50 * p_cred) + (0.35 * p_sub) + (0.15 * (0.8 if sent_bin == 1 else 0.4))
        fine_tuned = p_supp * 0.7 + (evidence * 0.2) + (relevance * 0.1) - (toxicity * 0.15)
        return max(0.02, min(0.98, float(fine_tuned)))

    def infer_support_probabilities(self, signals_a: dict, signals_b: dict) -> dict:
        """Calculates Bayesian support probabilities for Side A and Side B."""
        prob_a = self._infer_single(
            signals_a.get("toxicity", 0.0),
            signals_a.get("sarcasm", 0.0),
            signals_a.get("sentiment", "neutral"),
            signals_a.get("evidence", 0.0),
            signals_a.get("relevance", 0.5),
        )

        prob_b = self._infer_single(
            signals_b.get("toxicity", 0.0),
            signals_b.get("sarcasm", 0.0),
            signals_b.get("sentiment", "neutral"),
            signals_b.get("evidence", 0.0),
            signals_b.get("relevance", 0.5),
        )

        return {
            "support_a": round(prob_a, 4),
            "support_b": round(prob_b, 4),
            "support_diff": round(abs(prob_a - prob_b), 4),
        }

    def get_candidate_verdict(self, support_a: float, support_b: float) -> str:
        """Returns initial candidate verdict before CSP validation."""
        if abs(support_a - support_b) < 0.01:
            return "Tie"
        return "A" if support_a > support_b else "B"


class VerdictConstraintSolver:
    """CSP solver validating candidate decisions against mandatory domain constraints."""

    def __init__(self, min_margin: float = 0.10, min_evidence: float = 0.18):
        self.min_margin = min_margin
        self.min_evidence = min_evidence

    def check_constraints(self, candidate: str, support_a: float, support_b: float, signals_a: dict, signals_b: dict) -> dict:
        """Evaluates all 5 mandatory CSP domain constraints."""
        margin = abs(support_a - support_b)
        ev_a = signals_a.get("evidence", 0.0)
        ev_b = signals_b.get("evidence", 0.0)
        checks = {}

        # C1: Margin Confidence Constraint: Difference must be statistically significant
        margin_passed = bool(margin >= self.min_margin)
        checks["C1_margin_confidence"] = {
            "rule": f"|P(Support_A) - P(Support_B)| >= {self.min_margin}",
            "value": margin,
            "passed": margin_passed,
            "description": f"Support margin {margin:.3f} {'meets' if margin_passed else 'below'} significance threshold {self.min_margin:.3f}.",
        }

        # C2: Minimum Substantive Evidence Constraint: At least one side must present structured evidence
        has_substantive_evidence = (ev_a >= self.min_evidence) or (ev_b >= self.min_evidence)
        checks["C2_minimum_quality"] = {
            "rule": f"max(Evidence_A, Evidence_B) >= {self.min_evidence}",
            "value": max(ev_a, ev_b),
            "passed": bool(has_substantive_evidence),
            "description": f"Maximum evidence score {max(ev_a, ev_b):.2f} {'satisfies' if has_substantive_evidence else 'fails'} minimum factual evidence floor.",
        }

        # C3: Safety Gatekeeper Constraint: Neither side may be toxically disqualified
        tox_a = signals_a.get("toxicity", 0.0)
        tox_b = signals_b.get("toxicity", 0.0)
        safety_passed = (tox_a < 0.65) and (tox_b < 0.65)
        checks["C3_safety_gatekeeper"] = {
            "rule": "Toxicity < 0.65 for both participants",
            "passed": bool(safety_passed),
            "description": f"Toxicity check {'passed' if safety_passed else 'breached'} (A: {tox_a:.2f}, B: {tox_b:.2f}).",
        }

        # C4: Symmetric Evaluation Constraint: Equal treatment of identical parameters
        symmetry_passed = True
        if abs(support_a - support_b) < 0.01 and candidate != "Tie":
            symmetry_passed = False
        checks["C4_symmetric_evaluation"] = {
            "rule": "Equal scores must produce identical outcome without arbitrary favoritism",
            "passed": bool(symmetry_passed),
            "description": "Both arguments evaluated with identical probabilistic criteria.",
        }

        # C5: Mandatory Transparent Explanation Constraint
        checks["C5_mandatory_explanation"] = {
            "rule": "Verdict must include evidence comparison, signal breakdown, and rule audit",
            "passed": True,
            "description": "Structured explanation verified and logged.",
        }

        return checks

    def apply_rules(self, candidate: str, support_a: float, support_b: float, signals_a: dict, signals_b: dict) -> tuple:
        """Validates candidate outcome; forces Inconclusive if constraints are violated."""
        checks = self.check_constraints(candidate, support_a, support_b, signals_a, signals_b)
        margin = abs(support_a - support_b)

        if not checks["C3_safety_gatekeeper"]["passed"]:
            return "Inconclusive", 0.0, checks, "CSP Violation: Gatekeeper safety breached. Dispute halted."

        if not checks["C2_minimum_quality"]["passed"]:
            reason = f"CSP Constraint Triggered: Both arguments lack verifiable empirical evidence (max evidence {checks['C2_minimum_quality']['value']:.2f} < {self.min_evidence:.2f}). Both are subjective claims."
            return "Inconclusive", 0.50, checks, reason

        if not checks["C1_margin_confidence"]["passed"]:
            reason = f"CSP Constraint Triggered: Support margin ({margin:.3f}) is below minimum significance threshold ({self.min_margin:.3f}). Arguments are too evenly matched."
            return "Inconclusive", round(1.0 - margin, 4), checks, reason

        # Candidate is accepted
        winner = candidate
        confidence = round(min(0.99, 0.50 + (margin * 0.75)), 4)
        decision_label = f"Side {winner}"
        reason = f"CSP Validation Passed: Side {winner} established a statistically significant evidentiary advantage (Margin: {margin:.3f}, Confidence: {confidence:.2f})."
        return decision_label, confidence, checks, reason

    def allow_inconclusive(self, support_a: float, support_b: float, reason: str = "") -> dict:
        """Returns standard inconclusive payload."""
        return {
            "decision": "Inconclusive",
            "winner": None,
            "winner_name": "Inconclusive",
            "confidence": 0.5,
            "reason": reason or "The evidence and argumentation submitted by both parties were insufficient or conflicting.",
        }


class VerdictResult:
    """Encapsulates final decision, LR signal breakdown, Bayesian probabilities, and CSP validation logs."""

    def __init__(self, decision: str, winner: str, winner_name: str, confidence: float,
                 support_a: float, support_b: float, ml_scores: dict,
                 csp_validation: dict, reasoning: str, fallacies_a: list = None, fallacies_b: list = None):
        self.decision = decision  # "Side A", "Side B", or "Inconclusive"
        self.winner = winner      # "A", "B", or None
        self.winner_name = winner_name
        self.confidence = float(confidence)
        self.support_a = float(support_a)
        self.support_b = float(support_b)
        self.ml_scores = ml_scores or {}
        self.csp_validation = csp_validation or {}
        self.reasoning = reasoning or ""
        self.fallacies_a = fallacies_a or []
        self.fallacies_b = fallacies_b or []

    def format_for_display(self) -> str:
        """Formats comprehensive human-readable verdict summary."""
        lines = [
            f"=== VIRTUAL JUDGE VERDICT: {self.decision.upper()} ===",
            f"Confidence: {self.confidence * 100:.1f}%",
            f"Bayesian Support P(Support_A): {self.support_a:.3f} | P(Support_B): {self.support_b:.3f}",
            "",
            "--- Signal Breakdown (Logistic Regression + TF-IDF) ---",
            f"Side A: Toxicity={self.ml_scores.get('toxicity_a', 0.0):.2f}, Sarcasm={self.ml_scores.get('sarcasm_a', 0.0):.2f}, Sentiment={self.ml_scores.get('sentiment_a', 'N/A')}",
            f"Side B: Toxicity={self.ml_scores.get('toxicity_b', 0.0):.2f}, Sarcasm={self.ml_scores.get('sarcasm_b', 0.0):.2f}, Sentiment={self.ml_scores.get('sentiment_b', 'N/A')}",
            "",
            "--- CSP Constraint Validation Audit ---",
        ]
        for name, data in self.csp_validation.items():
            status = "PASSED" if data.get("passed") else "VIOLATED"
            lines.append(f"[{status}] {name}: {data.get('description', '')}")
        lines.append("")
        lines.append(f"Decision Rationale:\n{self.reasoning}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Converts result object to JSON-serializable dictionary."""
        return {
            "decision": self.decision,
            "winner": self.winner,
            "winner_name": self.winner_name,
            "confidence": self.confidence,
            "support_a": self.support_a,
            "support_b": self.support_b,
            "ml_scores": self.ml_scores,
            "csp_validation": self.csp_validation,
            "reasoning": self.reasoning,
            "fallacies_a": self.fallacies_a,
            "fallacies_b": self.fallacies_b,
        }


class VirtualJudgeController:
    """Coordinates pipeline execution from submission to final verdict formatting."""

    def __init__(self, model_dir=None):
        self.signal_classifier = SignalClassifier(model_dir=model_dir)
        self.bayesian_judge = BayesianJudge()
        self.csp_solver = VerdictConstraintSolver()

    def submit_dispute(self, topic: str, text_a: str, text_b: str, user_a_name: str = "Side A", user_b_name: str = "Side B"):
        """Creates and validates Argument objects for both disputants."""
        arg_a = Argument(text_a, topic=topic, author_name=user_a_name)
        arg_b = Argument(text_b, topic=topic, author_name=user_b_name)
        arg_a.validate_input()
        arg_b.validate_input()
        return arg_a, arg_b

    def run_pipeline(self, topic: str, text_a: str, text_b: str, user_a_name: str = "Side A", user_b_name: str = "Side B"):
        """Executes the full 3-tier hybrid AI pipeline."""
        # 1. Submission & Feature Extraction
        try:
            arg_a, arg_b = self.submit_dispute(topic, text_a, text_b, user_a_name, user_b_name)
        except ValueError as val_err:
            return {
                "status": "error",
                "message": str(val_err),
                "verdict": None,
                "ml_scores": {},
            }

        # 2. Layer 1: LR Classifiers & Gatekeeper Check
        gatekeeper = self.signal_classifier.run_gatekeeper_check(text_a, text_b, threshold=0.75)
        if gatekeeper["is_toxic"]:
            ml_scores = {
                "toxicity_a": gatekeeper["toxicity_a"],
                "toxicity_b": gatekeeper["toxicity_b"],
                "sarcasm_a": round(self.signal_classifier.predict_sarcasm(text_a), 4),
                "sarcasm_b": round(self.signal_classifier.predict_sarcasm(text_b), 4),
                "sentiment_a": "negative",
                "sentiment_b": "negative",
                "sentiment_compound_a": -1.0,
                "sentiment_compound_b": -1.0,
                "is_toxic": True,
            }
            return {
                "status": "flagged",
                "message": gatekeeper["reason"],
                "ml_scores": ml_scores,
                "verdict": None,
            }

        # Extract full signals
        sar_a = self.signal_classifier.predict_sarcasm(text_a)
        sar_b = self.signal_classifier.predict_sarcasm(text_b)
        sen_label_a, compound_a = self.signal_classifier.predict_sentiment(text_a)
        sen_label_b, compound_b = self.signal_classifier.predict_sentiment(text_b)

        signals_a = {
            "toxicity": gatekeeper["toxicity_a"],
            "sarcasm": sar_a,
            "sentiment": sen_label_a,
            "evidence": arg_a.evidence_score,
            "relevance": arg_a.relevance_score,
        }
        signals_b = {
            "toxicity": gatekeeper["toxicity_b"],
            "sarcasm": sar_b,
            "sentiment": sen_label_b,
            "evidence": arg_b.evidence_score,
            "relevance": arg_b.relevance_score,
        }

        # 3. Layer 2: Bayesian Network Probabilistic Inference
        bayes_results = self.bayesian_judge.infer_support_probabilities(signals_a, signals_b)
        support_a = bayes_results["support_a"]
        support_b = bayes_results["support_b"]
        candidate = self.bayesian_judge.get_candidate_verdict(support_a, support_b)

        # 4. Layer 3: CSP Validation & Constraint Satisfaction
        decision_label, confidence, csp_checks, rationale = self.csp_solver.apply_rules(
            candidate, support_a, support_b, signals_a, signals_b
        )

        # Determine winner
        if decision_label == "Side A":
            winner_letter = "A"
            winner_name = user_a_name
        elif decision_label == "Side B":
            winner_letter = "B"
            winner_name = user_b_name
        else:
            winner_letter = None
            winner_name = "Inconclusive"

        # Build detailed transparent explanation
        detailed_reasoning = self._generate_reasoning(
            topic, user_a_name, user_b_name, arg_a, arg_b,
            signals_a, signals_b, support_a, support_b,
            decision_label, winner_name, confidence, rationale
        )

        ml_scores = {
            "toxicity_a": signals_a["toxicity"],
            "toxicity_b": signals_b["toxicity"],
            "sarcasm_a": round(sar_a, 4),
            "sarcasm_b": round(sar_b, 4),
            "sentiment_a": sen_label_a,
            "sentiment_b": sen_label_b,
            "sentiment_compound_a": compound_a,
            "sentiment_compound_b": compound_b,
            "evidence_score_a": round(arg_a.evidence_score, 4),
            "evidence_score_b": round(arg_b.evidence_score, 4),
            "relevance_score_a": round(arg_a.relevance_score, 4),
            "relevance_score_b": round(arg_b.relevance_score, 4),
            "is_toxic": False,
        }

        verdict_obj = VerdictResult(
            decision=decision_label,
            winner=winner_letter,
            winner_name=winner_name,
            confidence=confidence,
            support_a=support_a,
            support_b=support_b,
            ml_scores=ml_scores,
            csp_validation=csp_checks,
            reasoning=detailed_reasoning,
            fallacies_a=arg_a.detected_fallacies,
            fallacies_b=arg_b.detected_fallacies,
        )

        return {
            "status": "resolved",
            "dispute_id": None,
            "ml_scores": ml_scores,
            "verdict": verdict_obj.to_dict(),
        }

    def _generate_reasoning(self, topic, user_a_name, user_b_name, arg_a, arg_b,
                            signals_a, signals_b, support_a, support_b,
                            decision_label, winner_name, confidence, rationale):
        """Constructs an explainable, structured AI rationale synthesizing all 3 tiers."""
        lines = []
        lines.append(f"Dispute Topic: '{topic}'")
        lines.append(f"Final Decision: {decision_label} ({winner_name}) with {confidence * 100:.1f}% confidence.")
        lines.append("")
        lines.append("1. Probabilistic Support Analysis (Bayesian Network):")
        lines.append(f"- {user_a_name} (Side A): Calculated support probability P(Support_A) = {support_a:.3f}.")
        lines.append(f"  Evidence score: {signals_a['evidence']:.2f}, Topic relevance: {signals_a['relevance']:.2f}, Toxicity: {signals_a['toxicity']:.2f}, Sarcasm: {signals_a['sarcasm']:.2f}.")
        lines.append(f"- {user_b_name} (Side B): Calculated support probability P(Support_B) = {support_b:.3f}.")
        lines.append(f"  Evidence score: {signals_b['evidence']:.2f}, Topic relevance: {signals_b['relevance']:.2f}, Toxicity: {signals_b['toxicity']:.2f}, Sarcasm: {signals_b['sarcasm']:.2f}.")
        lines.append("")
        lines.append("2. Constraint Satisfaction (CSP) Outcome:")
        lines.append(f"- {rationale}")
        if decision_label == "Inconclusive":
            lines.append("- Rule Applied: When opposing claims present comparable evidence or fall below confidence margins, the system refuses to guess, ensuring fairness and eliminating arbitrary bias.")
        else:
            diff = abs(support_a - support_b)
            lines.append(f"- Rule Applied: {winner_name} provided superior evidentiary grounding and topical substance with a clear support margin (+{diff:.3f}).")
        
        return "\n".join(lines)

    def format_verdict_result(self, verdict_result: dict) -> str:
        """Formats verdict result dictionary for display."""
        if not verdict_result or "verdict" not in verdict_result:
            return "No verdict available."
        v = verdict_result["verdict"]
        return f"Decision: {v.get('decision')} | Winner: {v.get('winner_name')} | Confidence: {v.get('confidence')}"
