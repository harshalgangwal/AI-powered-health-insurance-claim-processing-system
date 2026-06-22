"""
agents/fraud_detector.py
------------------------
Agent 4 — Fraud Detector

Queries SQLite claim history and applies rule-based fraud checks.
Returns a FraudResult that the orchestrator uses to potentially
upgrade the decision to MANUAL_REVIEW.

Checks performed (in order):
1. Same-day claim count for this member
2. Monthly claim count for this member
3. High-value claim threshold
4. Duplicate claim detection (same member + same treatment date + same amount)
5. OCR-flagged document alteration signals
6. Exact-round-number billing (minor heuristic)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from app.config import get_policy
from app.database import count_claims_today, count_claims_this_month
from app.models import AgentStatus, ExtractionResult, FraudResult

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class FraudDetector:
    """
    Stateless fraud detection agent.
    All thresholds are read from policy_terms.json via config.py.
    """

    def __init__(self) -> None:
        self._policy = get_policy()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        member_id: str,
        treatment_date: str,
        claimed_amount: float,
        extraction: ExtractionResult | None,
        db: Session,
    ) -> FraudResult:
        """
        Run all fraud checks and return a FraudResult.

        Args:
            member_id:       Member ID from the claim submission.
            treatment_date:  ISO date string of the treatment.
            claimed_amount:  Amount as submitted by the member.
            extraction:      Output of Agent 2 (may be None if Agent 2 failed).
            db:              Active SQLAlchemy session.

        Returns:
            FraudResult with fraud_score, flags, and override_to_manual.
        """
        flags: list[str] = []
        fraud_score = 0.0

        # ── 1. Same-day claim count ─────────────────────────────────────
        same_day_count = count_claims_today(db, member_id, treatment_date)
        if same_day_count >= self._policy.fraud_same_day_limit:
            flags.append(
                f"SAME_DAY_LIMIT_EXCEEDED: {same_day_count + 1} claims on {treatment_date} "
                f"(limit {self._policy.fraud_same_day_limit})"
            )
            fraud_score = min(fraud_score + 0.40, 1.0)
        elif same_day_count >= self._policy.fraud_same_day_limit - 1:
            # Approaching the limit — mild signal
            flags.append(
                f"SAME_DAY_CLAIM_COUNT_HIGH: {same_day_count + 1} claims on {treatment_date}"
            )
            fraud_score = min(fraud_score + 0.15, 1.0)

        # ── 2. Monthly claim count ──────────────────────────────────────
        monthly_count = count_claims_this_month(db, member_id, treatment_date)
        if monthly_count >= self._policy.fraud_monthly_limit:
            flags.append(
                f"MONTHLY_LIMIT_EXCEEDED: {monthly_count + 1} claims this month "
                f"(limit {self._policy.fraud_monthly_limit})"
            )
            fraud_score = min(fraud_score + 0.30, 1.0)

        # ── 3. High-value claim ─────────────────────────────────────────
        if claimed_amount >= self._policy.fraud_high_value_threshold:
            flags.append(
                f"HIGH_VALUE_CLAIM: ₹{claimed_amount:,.0f} "
                f"≥ threshold ₹{self._policy.fraud_high_value_threshold:,.0f}"
            )
            fraud_score = min(fraud_score + 0.20, 1.0)

        # ── 4. Duplicate detection ──────────────────────────────────────
        # Checked via same-day count above; if same day AND same amount
        # that's a stronger duplicate signal.
        if same_day_count >= 1 and extraction is not None:
            extracted_total = extraction.total_amount
            if extracted_total is not None and abs(extracted_total - claimed_amount) < 1.0:
                flags.append(
                    "POSSIBLE_DUPLICATE: same member, same date, same amount as an existing claim"
                )
                fraud_score = min(fraud_score + 0.25, 1.0)

        # ── 5. Document alteration signals from OCR ─────────────────────
        if extraction is not None:
            for doc in extraction.documents:
                if "DOCUMENT_ALTERATION" in doc.flags:
                    flags.append(
                        f"DOCUMENT_ALTERATION_DETECTED: file {doc.file_id} "
                        "flagged by OCR extractor"
                    )
                    fraud_score = min(fraud_score + 0.35, 1.0)
                if doc.ocr_confidence < 0.40 and doc.low_confidence_fields:
                    flags.append(
                        f"LOW_OCR_CONFIDENCE: file {doc.file_id} "
                        f"confidence {doc.ocr_confidence:.0%}, "
                        f"suspicious fields: {', '.join(doc.low_confidence_fields[:3])}"
                    )
                    fraud_score = min(fraud_score + 0.10, 1.0)

        # ── 6. Round-number billing heuristic ───────────────────────────
        # Perfectly round numbers (multiples of 500) can indicate fabricated bills.
        if claimed_amount >= 1000 and claimed_amount % 500 == 0:
            flags.append(
                f"ROUND_NUMBER_BILLING: claimed amount ₹{claimed_amount:,.0f} "
                "is suspiciously round"
            )
            fraud_score = min(fraud_score + 0.05, 1.0)

        # ── Escalation decision ─────────────────────────────────────────
        override_to_manual = (
            fraud_score >= self._policy.fraud_score_threshold
            or claimed_amount >= self._policy.fraud_auto_manual_review_above
        )

        if override_to_manual and not flags:
            # Amount alone triggered manual review
            flags.append(
                f"AUTO_MANUAL_REVIEW: amount ₹{claimed_amount:,.0f} "
                f"exceeds auto-review threshold ₹{self._policy.fraud_auto_manual_review_above:,.0f}"
            )

        logger.info(
            "FraudDetector: member=%s score=%.2f flags=%d override=%s",
            member_id,
            fraud_score,
            len(flags),
            override_to_manual,
        )

        return FraudResult(
            fraud_score=round(fraud_score, 4),
            flags=flags,
            override_to_manual=override_to_manual,
            same_day_claim_count=same_day_count,
            monthly_claim_count=monthly_count,
            agent_status=AgentStatus.SUCCESS,
        )
