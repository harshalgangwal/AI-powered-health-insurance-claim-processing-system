"""
policy_engine.py
----------------
Agent 3 — Policy Engine

Pure Python adjudication. Every rule value is read from PolicyConfig (which
reads policy_terms.json). Nothing is hardcoded here.

Checks run in this order — each produces a CheckResult added to the trace:
  1.  member_exists          — member_id in policy roster
  2.  policy_active          — policy renewal_status == ACTIVE
  3.  claim_within_deadline  — treatment_date within submission_deadline_days
  4.  minimum_claim_amount   — claimed_amount >= minimum_claim_amount
  5.  initial_waiting_period — join_date + 30 days <= treatment_date
  6.  condition_waiting_period — diagnosis-specific waiting period
  7.  global_exclusion       — diagnosis/treatment not in exclusions list
  8.  pre_authorization      — pre-auth required and present (or not needed)
  9.  per_claim_limit        — claimed_amount <= per_claim_limit
 10.  category_sub_limit     — claimed_amount <= category sub_limit
 11.  annual_opd_limit       — ytd_amount + claimed <= annual_opd_limit
 12.  line_item_adjudication — approve/reject each line item individually
 13.  financial_calculation  — network discount → co-pay → approved amount

The engine stops accumulating checks as soon as a HARD STOP condition is met
(member not found, global exclusion, waiting period) but always records WHY
it stopped. Soft limits (sub-limit, annual limit) result in PARTIAL decisions
rather than outright rejection.

Public interface:
    engine = PolicyEngine()
    decision: PolicyDecision = engine.adjudicate(
        submission, extraction_result, ytd_amount, db_session
    )
"""

from __future__ import annotations

import logging
import re
from datetime import date, timedelta
from typing import Any

from app.config import get_policy
from app.models import (
    AgentStatus,
    CheckResult,
    ClaimCategory,
    ClaimSubmitRequest,
    DecisionType,
    ExtractionResult,
    LineItem,
    PolicyDecision,
)

logger = logging.getLogger(__name__)

# Sentinel for "check was not run because an earlier check hard-stopped"
_SKIPPED = "SKIPPED"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_date(s: str | None) -> date | None:
    """Parse ISO date string 'YYYY-MM-DD' to a date object. Returns None on failure."""
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except (ValueError, TypeError):
        return None


def _format_inr(amount: float) -> str:
    """Format a float as an Indian Rupee string: ₹1,500.00"""
    return f"₹{amount:,.2f}"


def _diagnosis_matches_condition(diagnosis: str, condition_keyword: str) -> bool:
    """
    Case-insensitive check: does the diagnosis mention this condition?
    Handles common shorthand:
      T2DM / DM / diabetes mellitus  → "diabetes"
      HTN / hypertension             → "hypertension"
      hypothyroid / thyroid          → "thyroid_disorders"
    """
    if not diagnosis:
        return False

    d = diagnosis.lower()
    c = condition_keyword.lower().replace("_", " ")

    # Direct substring match
    if c in d:
        return True

    # Shorthand expansions
    shorthand_map: dict[str, list[str]] = {
        "diabetes": ["t2dm", "dm", "diabetes mellitus", "diabetic", "hyperglycaemia",
                     "hyperglycemia", "type 2 diabetes", "type ii diabetes"],
        "hypertension": ["htn", "high blood pressure", "hypertensive"],
        "thyroid_disorders": ["hypothyroid", "hyperthyroid", "thyroiditis", "thyroid"],
        "joint_replacement": ["knee replacement", "hip replacement", "arthroplasty"],
        "maternity": ["pregnancy", "antenatal", "prenatal", "obstetric", "delivery",
                      "labour", "labor", "caesarean", "c-section"],
        "mental_health": ["depression", "anxiety", "psychiatric", "psychosis",
                          "schizophrenia", "bipolar", "ocd", "ptsd"],
        "obesity_treatment": ["obesity", "bariatric", "weight loss", "morbid obesity",
                              "bmi", "overweight"],
        "hernia": ["hernia", "herniation"],
        "cataract": ["cataract"],
    }

    for keyword, synonyms in shorthand_map.items():
        if c in keyword or keyword in c:
            if any(syn in d for syn in synonyms):
                return True

    return False


def _text_matches_exclusion(text: str, exclusion: str) -> bool:
    """
    Check whether `text` (diagnosis or treatment description) triggers an
    exclusion. Both are lowercased; significant words (>3 chars) must all appear.
    """
    if not text or not exclusion:
        return False

    text_lower = text.lower()
    excl_lower = exclusion.lower()

    # Direct substring
    if excl_lower in text_lower:
        return True

    # All significant words present
    significant = [w for w in excl_lower.split() if len(w) > 3]
    if significant and all(w in text_lower for w in significant):
        return True

    return False


def _line_item_is_excluded(description: str, excluded_list: list[str]) -> tuple[bool, str]:
    """
    Returns (True, matching_exclusion) if the line item description matches
    any exclusion. Returns (False, "") otherwise.
    """
    for excl in excluded_list:
        if _text_matches_exclusion(description, excl):
            return True, excl
    return False, ""


def _line_item_is_covered(description: str, covered_list: list[str]) -> bool:
    """
    Returns True if the description matches any entry in the covered list.
    When a covered list exists, items NOT in it are treated as not covered.
    """
    if not covered_list:
        # No explicit covered list → all items covered unless excluded
        return True
    desc_lower = description.lower()
    for item in covered_list:
        if _text_matches_exclusion(desc_lower, item):
            return True
    return False


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class PolicyEngine:
    """
    Agent 3 — Policy Engine.

    Stateless: every call to adjudicate() is independent.
    All policy values come from PolicyConfig at call time (no caching here).
    """

    def __init__(self) -> None:
        self._policy = get_policy()

    def adjudicate(
        self,
        submission:    ClaimSubmitRequest,
        extraction:    ExtractionResult,
        ytd_amount:    float = 0.0,
        pre_auth_present: bool = False,
    ) -> PolicyDecision:
        """
        Run the full adjudication sequence and return a PolicyDecision.

        Args:
            submission:       Parsed claim submission from the HTTP request.
            extraction:       Output of Agent 2 (OCR extractor).
            ytd_amount:       Year-to-date approved amount for this member
                              (queried from DB by orchestrator, or provided in test).
            pre_auth_present: Whether a valid pre-authorisation exists for this claim.

        Returns:
            PolicyDecision with decision, amounts, full check trace, and confidence.
        """
        try:
            return self._run_adjudication(submission, extraction, ytd_amount, pre_auth_present)
        except Exception as exc:
            logger.exception("PolicyEngine crashed: %s", exc)
            return PolicyDecision(
                decision     = DecisionType.MANUAL_REVIEW,
                claimed_amount = submission.claimed_amount,
                checks       = [CheckResult(
                    check_name = "engine_error",
                    passed     = False,
                    detail     = f"Policy engine encountered an unexpected error: {exc}",
                )],
                confidence   = 0.0,
                agent_status = AgentStatus.FAILED,
                agent_error  = str(exc),
            )

    def _run_adjudication(
        self,
        submission:       ClaimSubmitRequest,
        extraction:       ExtractionResult,
        ytd_amount:       float,
        pre_auth_present: bool,
    ) -> PolicyDecision:

        p          = self._policy
        checks:    list[CheckResult] = []
        category   = submission.claim_category.value  # e.g. "CONSULTATION"

        # Use YTD from submission if provided (test harness), else from DB query result
        effective_ytd = submission.ytd_claims_amount if submission.ytd_claims_amount > 0 else ytd_amount

        # ── Resolve diagnosis: prefer extraction, fall back to nothing ────────
        diagnosis = extraction.diagnosis or ""
        hospital_name_on_bill = (
            extraction.hospital_name
            or submission.hospital_name
            or ""
        )
        # Claimed amount: use submission value (user-declared);
        # extraction total is used for cross-validation only
        claimed = submission.claimed_amount

        # =========================================================
        # CHECK 1: Member exists
        # =========================================================
        member = p.get_member(submission.member_id)
        if member is None:
            checks.append(CheckResult(
                check_name = "member_exists",
                passed     = False,
                detail     = (
                    f"Member ID '{submission.member_id}' was not found in the policy roster. "
                    f"Please verify the member ID and resubmit."
                ),
            ))
            return self._build_decision(
                DecisionType.REJECTED, 0.0, claimed, checks,
                ["MEMBER_NOT_FOUND"], 0.0, 0.0, confidence=0.95,
            )

        checks.append(CheckResult(
            check_name = "member_exists",
            passed     = True,
            detail     = f"Member '{member['name']}' (ID: {submission.member_id}) found in roster.",
        ))

        # =========================================================
        # CHECK 2: Policy active
        # =========================================================
        renewal = p.raw().get("policy_holder", {}).get("renewal_status", "UNKNOWN")
        policy_active = renewal == "ACTIVE"
        checks.append(CheckResult(
            check_name = "policy_active",
            passed     = policy_active,
            detail     = f"Policy renewal status: {renewal}.",
        ))
        if not policy_active:
            return self._build_decision(
                DecisionType.REJECTED, 0.0, claimed, checks,
                ["POLICY_INACTIVE"], 0.0, 0.0, confidence=0.99,
            )

        # =========================================================
        # CHECK 3: Submission deadline
        # =========================================================
        treatment_dt = _parse_date(submission.treatment_date)
        today        = date.today()
        deadline_ok  = True
        if treatment_dt:
            days_since = (today - treatment_dt).days
            deadline_ok = days_since <= p.submission_deadline_days
            checks.append(CheckResult(
                check_name = "submission_deadline",
                passed     = deadline_ok,
                detail     = (
                    f"Treatment date: {submission.treatment_date}. "
                    f"Days since treatment: {days_since}. "
                    f"Deadline: {p.submission_deadline_days} days."
                ),
                impact     = None if deadline_ok else "Claim submitted after 30-day deadline.",
            ))
            if not deadline_ok:
                return self._build_decision(
                    DecisionType.REJECTED, 0.0, claimed, checks,
                    ["SUBMISSION_DEADLINE_EXCEEDED"], 0.0, 0.0, confidence=0.99,
                )
        else:
            checks.append(CheckResult(
                check_name = "submission_deadline",
                passed     = True,
                detail     = "Treatment date not parsed — deadline check skipped.",
            ))

        # =========================================================
        # CHECK 4: Minimum claim amount
        # =========================================================
        min_amount = p.minimum_claim_amount
        amount_ok  = claimed >= min_amount
        checks.append(CheckResult(
            check_name = "minimum_claim_amount",
            passed     = amount_ok,
            detail     = (
                f"Claimed amount {_format_inr(claimed)}. "
                f"Minimum claimable: {_format_inr(min_amount)}."
            ),
        ))
        if not amount_ok:
            return self._build_decision(
                DecisionType.REJECTED, 0.0, claimed, checks,
                ["BELOW_MINIMUM_CLAIM_AMOUNT"], 0.0, 0.0, confidence=0.99,
            )

        # =========================================================
        # CHECK 5: Initial waiting period (30 days from join date)
        # =========================================================
        join_dt    = _parse_date(member.get("join_date"))
        initial_ok = True
        if join_dt and treatment_dt:
            eligible_from = join_dt + timedelta(days=p.initial_waiting_period_days)
            initial_ok    = treatment_dt >= eligible_from
            checks.append(CheckResult(
                check_name = "initial_waiting_period",
                passed     = initial_ok,
                detail     = (
                    f"Member join date: {member['join_date']}. "
                    f"Initial waiting period: {p.initial_waiting_period_days} days. "
                    f"Eligible from: {eligible_from.isoformat()}. "
                    f"Treatment date: {submission.treatment_date}."
                ),
                impact     = (
                    None if initial_ok
                    else f"Member not yet eligible. Eligible from {eligible_from.isoformat()}."
                ),
            ))
            if not initial_ok:
                return self._build_decision(
                    DecisionType.REJECTED, 0.0, claimed, checks,
                    ["INITIAL_WAITING_PERIOD"], 0.0, 0.0, confidence=0.99,
                )
        else:
            checks.append(CheckResult(
                check_name = "initial_waiting_period",
                passed     = True,
                detail     = "Could not parse dates — initial waiting period check skipped.",
            ))

        # =========================================================
        # CHECK 6: Condition-specific waiting period  (TC005)
        # =========================================================
        condition_waiting_ok = True
        if diagnosis and join_dt and treatment_dt:
            specific_conditions = p.raw().get("waiting_periods", {}).get("specific_conditions", {})
            matched_condition   = None
            waiting_days        = None

            for condition, days in specific_conditions.items():
                if _diagnosis_matches_condition(diagnosis, condition):
                    matched_condition = condition
                    waiting_days      = int(days)
                    break

            if matched_condition and waiting_days is not None:
                eligible_from      = join_dt + timedelta(days=waiting_days)
                condition_waiting_ok = treatment_dt >= eligible_from
                checks.append(CheckResult(
                    check_name = "condition_waiting_period",
                    passed     = condition_waiting_ok,
                    detail     = (
                        f"Diagnosis '{diagnosis}' matches condition '{matched_condition}' "
                        f"which has a {waiting_days}-day waiting period. "
                        f"Member join date: {member['join_date']}. "
                        f"Eligible from: {eligible_from.isoformat()}. "
                        f"Treatment date: {submission.treatment_date}."
                    ),
                    impact     = (
                        None if condition_waiting_ok
                        else (
                            f"Waiting period not met. "
                            f"Member will be eligible for {matched_condition}-related claims "
                            f"from {eligible_from.isoformat()}."
                        )
                    ),
                ))
                if not condition_waiting_ok:
                    return self._build_decision(
                        DecisionType.REJECTED, 0.0, claimed, checks,
                        ["WAITING_PERIOD"], 0.0, 0.0, confidence=0.99,
                    )
            else:
                checks.append(CheckResult(
                    check_name = "condition_waiting_period",
                    passed     = True,
                    detail     = (
                        f"Diagnosis '{diagnosis}' has no specific waiting period. "
                        f"Standard waiting period check passed."
                        if diagnosis else
                        "No diagnosis extracted — condition waiting period check skipped."
                    ),
                ))
        else:
            checks.append(CheckResult(
                check_name = "condition_waiting_period",
                passed     = True,
                detail     = "Skipped — diagnosis or dates not available.",
            ))

        # =========================================================
        # CHECK 7: Global exclusions  (TC012)
        # =========================================================
        exclusion_triggered = False
        matched_exclusion   = ""

        # Check diagnosis
        if diagnosis:
            excl, reason = p.is_excluded(diagnosis)
            if excl:
                exclusion_triggered = True
                matched_exclusion   = reason

        # Also check any treatment description fields from extraction
        if not exclusion_triggered:
            for doc in extraction.documents:
                for li in doc.line_items:
                    excl, reason = p.is_excluded(li.description)
                    if excl:
                        exclusion_triggered = True
                        matched_exclusion   = reason
                        break
                if exclusion_triggered:
                    break

        checks.append(CheckResult(
            check_name = "global_exclusion",
            passed     = not exclusion_triggered,
            detail     = (
                f"Diagnosis/treatment '{diagnosis}' matches policy exclusion: '{matched_exclusion}'."
                if exclusion_triggered else
                f"No exclusions triggered for diagnosis: '{diagnosis or 'not extracted'}'."
            ),
            impact     = "Claim rejected — excluded condition." if exclusion_triggered else None,
        ))
        if exclusion_triggered:
            return self._build_decision(
                DecisionType.REJECTED, 0.0, claimed, checks,
                ["EXCLUDED_CONDITION"], 0.0, 0.0, confidence=0.95,
            )

        # =========================================================
        # CHECK 8: Pre-authorisation  (TC007)
        # =========================================================
        pre_auth_required = self._check_pre_auth_required(
            category, claimed, diagnosis, extraction
        )
        if pre_auth_required and not pre_auth_present:
            checks.append(CheckResult(
                check_name = "pre_authorization",
                passed     = False,
                detail     = (
                    f"Pre-authorisation is required for this claim "
                    f"({category}, {_format_inr(claimed)}) but was not provided. "
                    f"Claims requiring pre-auth: "
                    f"{', '.join(p.get_pre_auth_requirements())}."
                ),
                impact     = "Claim rejected — missing pre-authorisation.",
            ))
            return self._build_decision(
                DecisionType.REJECTED, 0.0, claimed, checks,
                ["PRE_AUTH_REQUIRED"], 0.0, 0.0, confidence=0.99,
            )
        else:
            checks.append(CheckResult(
                check_name = "pre_authorization",
                passed     = True,
                detail     = (
                    "Pre-authorisation required and present."
                    if pre_auth_required and pre_auth_present else
                    "Pre-authorisation not required for this claim."
                ),
            ))

        # =========================================================
        # CHECK 9: Per-claim hard limit  (TC008)
        # =========================================================
        per_claim_limit = p.per_claim_limit
        per_claim_ok    = claimed <= per_claim_limit
        checks.append(CheckResult(
            check_name = "per_claim_limit",
            passed     = per_claim_ok,
            detail     = (
                f"Claimed amount: {_format_inr(claimed)}. "
                f"Per-claim limit: {_format_inr(per_claim_limit)}."
            ),
            impact     = (
                None if per_claim_ok else
                f"Claimed {_format_inr(claimed)} exceeds the per-claim limit of "
                f"{_format_inr(per_claim_limit)}. Claim rejected."
            ),
        ))
        if not per_claim_ok:
            return self._build_decision(
                DecisionType.REJECTED, 0.0, claimed, checks,
                ["PER_CLAIM_EXCEEDED"], 0.0, 0.0, confidence=0.99,
            )

        # =========================================================
        # CHECK 10: Category sub-limit
        # =========================================================
        sub_limit    = p.get_sub_limit(category)
        sub_limit_ok = True
        capped_amount = claimed  # may be reduced if sub-limit is lower

        if sub_limit is not None:
            sub_limit_ok  = claimed <= sub_limit
            capped_amount = min(claimed, sub_limit)
            checks.append(CheckResult(
                check_name = "category_sub_limit",
                passed     = sub_limit_ok,
                detail     = (
                    f"Category '{category}' sub-limit: {_format_inr(sub_limit)}. "
                    f"Claimed: {_format_inr(claimed)}."
                ),
                impact     = (
                    None if sub_limit_ok else
                    f"Claim capped at category sub-limit {_format_inr(sub_limit)}. "
                    f"Excess {_format_inr(claimed - sub_limit)} not payable."
                ),
            ))
        else:
            checks.append(CheckResult(
                check_name = "category_sub_limit",
                passed     = True,
                detail     = f"No sub-limit defined for category '{category}'.",
            ))

        # =========================================================
        # CHECK 11: Annual OPD limit
        # =========================================================
        annual_limit    = p.annual_opd_limit
        projected_total = effective_ytd + capped_amount
        annual_ok       = projected_total <= annual_limit
        annual_headroom = max(0.0, annual_limit - effective_ytd)

        if not annual_ok:
            capped_amount = annual_headroom
            checks.append(CheckResult(
                check_name = "annual_opd_limit",
                passed     = False,
                detail     = (
                    f"Annual OPD limit: {_format_inr(annual_limit)}. "
                    f"YTD approved: {_format_inr(effective_ytd)}. "
                    f"Remaining headroom: {_format_inr(annual_headroom)}. "
                    f"Claim capped at {_format_inr(annual_headroom)}."
                ),
                impact     = (
                    f"Annual OPD limit reached. "
                    f"Only {_format_inr(annual_headroom)} payable from this claim."
                    if annual_headroom > 0 else
                    "Annual OPD limit fully exhausted. No amount payable."
                ),
            ))
            if annual_headroom <= 0:
                return self._build_decision(
                    DecisionType.REJECTED, 0.0, claimed, checks,
                    ["ANNUAL_LIMIT_EXHAUSTED"], 0.0, 0.0, confidence=0.99,
                )
        else:
            checks.append(CheckResult(
                check_name = "annual_opd_limit",
                passed     = True,
                detail     = (
                    f"Annual OPD limit: {_format_inr(annual_limit)}. "
                    f"YTD approved: {_format_inr(effective_ytd)}. "
                    f"Projected total after this claim: {_format_inr(projected_total)}. "
                    f"Within limit."
                ),
            ))

        # =========================================================
        # CHECK 12: Line-item adjudication  (TC006 dental, TC010 itemised)
        # =========================================================
        line_item_decisions, line_item_detail = self._adjudicate_line_items(
            extraction, category
        )
        if line_item_decisions:
            approved_items_total = sum(
                li.amount for li in line_item_decisions if li.included
            )
            rejected_items_total = sum(
                li.amount for li in line_item_decisions if not li.included
            )
            all_items_approved   = rejected_items_total == 0

            # If we have explicit line items, use their approved total instead
            # of the capped_amount from the sub-limit/annual checks.
            # But still respect the cap from earlier checks.
            line_item_approved = min(approved_items_total, capped_amount)

            checks.append(CheckResult(
                check_name = "line_item_adjudication",
                passed     = all_items_approved,
                detail     = line_item_detail,
                impact     = (
                    None if all_items_approved else
                    f"Approved: {_format_inr(line_item_approved)}, "
                    f"Rejected: {_format_inr(rejected_items_total)}."
                ),
            ))
            capped_amount = line_item_approved
        else:
            checks.append(CheckResult(
                check_name = "line_item_adjudication",
                passed     = True,
                detail     = "No itemised line items available — bulk amount adjudicated.",
            ))

        # =========================================================
        # CHECK 13: Financial calculation — discount → co-pay  (TC004, TC010)
        # =========================================================
        is_network    = p.is_network_hospital(hospital_name_on_bill)
        net_discount  = p.get_network_discount_percent(category) if is_network else 0.0
        copay_percent = p.get_copay_percent(category)

        # Step 1: Apply network discount to the approved amount before co-pay
        discount_amount = round(capped_amount * (net_discount / 100.0), 2)
        after_discount  = capped_amount - discount_amount

        # Step 2: Apply co-pay to the post-discount amount
        copay_amount    = round(after_discount * (copay_percent / 100.0), 2)
        final_amount    = round(after_discount - copay_amount, 2)

        financial_detail = (
            f"Base approvable amount: {_format_inr(capped_amount)}. "
        )
        if is_network and net_discount > 0:
            financial_detail += (
                f"Network hospital '{hospital_name_on_bill}' — "
                f"{net_discount:.0f}% discount applied: -{_format_inr(discount_amount)}. "
                f"After discount: {_format_inr(after_discount)}. "
            )
        else:
            financial_detail += (
                f"Hospital '{hospital_name_on_bill or 'unknown'}' is "
                f"{'a network hospital (0% discount for this category)' if is_network else 'not a network hospital — no discount'}. "
            )
        if copay_percent > 0:
            financial_detail += (
                f"Co-pay {copay_percent:.0f}% applied: -{_format_inr(copay_amount)}. "
                f"Final approved amount: {_format_inr(final_amount)}."
            )
        else:
            financial_detail += f"No co-pay for category '{category}'. Approved: {_format_inr(final_amount)}."

        checks.append(CheckResult(
            check_name = "financial_calculation",
            passed     = True,
            detail     = financial_detail,
            impact     = (
                f"Network discount: -{_format_inr(discount_amount)}. "
                f"Co-pay: -{_format_inr(copay_amount)}. "
                f"Approved: {_format_inr(final_amount)}."
                if (discount_amount > 0 or copay_amount > 0) else None
            ),
        ))

        # =========================================================
        # Final decision
        # =========================================================
        # Determine if this is a full approval, partial, or no payment
        if final_amount <= 0:
            final_decision     = DecisionType.REJECTED
            rejection_reasons  = ["ZERO_PAYABLE_AFTER_CALCULATIONS"]
        elif final_amount < claimed:
            final_decision     = DecisionType.PARTIAL
            rejection_reasons  = []
        else:
            final_decision     = DecisionType.APPROVED
            rejection_reasons  = []

        # Confidence: reduced if extraction confidence was low
        base_confidence = 0.95
        if extraction.overall_confidence < 0.60:
            base_confidence -= 0.15
        elif extraction.overall_confidence < 0.80:
            base_confidence -= 0.05

        return self._build_decision(
            decision          = final_decision,
            approved_amount   = final_amount,
            claimed_amount    = claimed,
            checks            = checks,
            rejection_reasons = rejection_reasons,
            copay_applied     = copay_amount,
            discount_applied  = discount_amount,
            confidence        = round(base_confidence, 3),
            line_item_decisions = line_item_decisions,
        )

    # -------------------------------------------------------------------------
    # Pre-auth check helper
    # -------------------------------------------------------------------------

    def _check_pre_auth_required(
        self,
        category:   str,
        claimed:    float,
        diagnosis:  str,
        extraction: ExtractionResult,
    ) -> bool:
        """
        Returns True if this claim requires pre-authorisation.
        TC007: MRI / CT Scan above ₹10,000 threshold requires pre-auth.
        """
        p = self._policy

        # Check if category itself requires pre-auth (e.g. inpatient)
        if p.requires_pre_auth(category):
            return True

        # Threshold-based: high-value diagnostic tests
        if category.upper() == "DIAGNOSTIC":
            threshold     = p.pre_auth_threshold(category)
            high_val_tests = p.high_value_tests_requiring_pre_auth(category)

            if threshold is not None and claimed > threshold and high_val_tests:
                # Check if any line item or diagnosis mentions a high-value test
                all_text = diagnosis.lower()
                for doc in extraction.documents:
                    for li in doc.line_items:
                        all_text += " " + li.description.lower()

                for test in high_val_tests:
                    if test.lower() in all_text:
                        return True

                # Also check extracted text for MRI/CT even without line items
                if any(t.lower() in all_text for t in high_val_tests):
                    return True

        # Check global pre-auth required_for list
        for requirement in p.get_pre_auth_requirements():
            req_lower = requirement.lower()
            # Format in policy: "MRI scan (amount > ₹10,000)"
            # Extract the test name and threshold if present
            amount_match = re.search(r">\s*[₹rs\.]*\s*([\d,]+)", req_lower)
            req_threshold = None
            if amount_match:
                try:
                    req_threshold = float(amount_match.group(1).replace(",", ""))
                except ValueError:
                    pass

            # Extract test name (everything before the parenthesis)
            test_name = re.sub(r"\s*\(.*?\)", "", req_lower).strip()

            if test_name:
                all_text = (diagnosis + " " + category).lower()
                for doc in extraction.documents:
                    for li in doc.line_items:
                        all_text += " " + li.description.lower()

                if test_name in all_text:
                    if req_threshold is None or claimed > req_threshold:
                        return True

        return False

    # -------------------------------------------------------------------------
    # Line-item adjudication helper
    # -------------------------------------------------------------------------

    def _adjudicate_line_items(
        self,
        extraction: ExtractionResult,
        category:   str,
    ) -> tuple[list[LineItem], str]:
        """
        Approve or reject each line item based on covered/excluded procedure lists.

        Returns:
            (list[LineItem with included/exclusion_reason set], summary_detail_string)
        """
        p = self._policy

        # Collect all line items from all extracted documents
        all_items: list[LineItem] = []
        for doc in extraction.documents:
            all_items.extend(doc.line_items)

        if not all_items:
            return [], ""

        covered_list  = p.get_covered_procedures(category)
        excluded_list = p.get_excluded_procedures(category)

        # Also get global exclusions for cross-checking
        global_excls  = p.get_global_exclusions()

        adjudicated: list[LineItem] = []
        detail_parts: list[str]     = []

        for item in all_items:
            # Check global exclusions first
            is_globally_excluded, global_reason = _line_item_is_excluded(
                item.description, global_excls
            )
            # Check category-specific exclusions
            is_cat_excluded, cat_reason = _line_item_is_excluded(
                item.description, excluded_list
            )
            # Check if in covered list (only enforced when covered_list is non-empty)
            is_covered = _line_item_is_covered(item.description, covered_list)

            if is_globally_excluded:
                adjudicated.append(LineItem(
                    description      = item.description,
                    amount           = item.amount,
                    included         = False,
                    exclusion_reason = f"Policy exclusion: {global_reason}",
                ))
                detail_parts.append(
                    f"  ✗ {item.description} ({_format_inr(item.amount)}): "
                    f"excluded — {global_reason}"
                )
            elif is_cat_excluded:
                adjudicated.append(LineItem(
                    description      = item.description,
                    amount           = item.amount,
                    included         = False,
                    exclusion_reason = f"Category exclusion: {cat_reason}",
                ))
                detail_parts.append(
                    f"  ✗ {item.description} ({_format_inr(item.amount)}): "
                    f"excluded — {cat_reason}"
                )
            elif covered_list and not is_covered:
                adjudicated.append(LineItem(
                    description      = item.description,
                    amount           = item.amount,
                    included         = False,
                    exclusion_reason = f"Not in covered procedures for {category}",
                ))
                detail_parts.append(
                    f"  ✗ {item.description} ({_format_inr(item.amount)}): "
                    f"not covered under {category}"
                )
            else:
                adjudicated.append(LineItem(
                    description      = item.description,
                    amount           = item.amount,
                    included         = True,
                    exclusion_reason = None,
                ))
                detail_parts.append(
                    f"  ✓ {item.description} ({_format_inr(item.amount)}): approved"
                )

        summary = (
            f"Line-item adjudication ({len(adjudicated)} items):\n"
            + "\n".join(detail_parts)
        )
        return adjudicated, summary

    # -------------------------------------------------------------------------
    # Decision builder
    # -------------------------------------------------------------------------

    @staticmethod
    def _build_decision(
        decision:           DecisionType,
        approved_amount:    float,
        claimed_amount:     float,
        checks:             list[CheckResult],
        rejection_reasons:  list[str],
        copay_applied:      float,
        discount_applied:   float,
        confidence:         float = 0.90,
        line_item_decisions: list[LineItem] | None = None,
    ) -> PolicyDecision:
        return PolicyDecision(
            decision             = decision,
            approved_amount      = round(approved_amount, 2),
            claimed_amount       = claimed_amount,
            copay_applied        = round(copay_applied, 2),
            discount_applied     = round(discount_applied, 2),
            rejection_reasons    = rejection_reasons,
            line_item_decisions  = line_item_decisions or [],
            checks               = checks,
            confidence           = confidence,
            agent_status         = AgentStatus.SUCCESS,
        )
