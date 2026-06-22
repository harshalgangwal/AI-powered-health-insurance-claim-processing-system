"""
config.py
---------
Loads policy_terms.json once at startup and exposes typed accessors.
All policy logic reads from here — nothing is hardcoded in agents.

Usage:
    from app.config import policy

    member   = policy.get_member("EMP001")
    rules    = policy.get_category_rules("CONSULTATION")
    days     = policy.get_waiting_period_days("diabetes")
    is_net   = policy.is_network_hospital("Apollo Hospitals")
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Settings (read from environment, with sensible defaults)
# ---------------------------------------------------------------------------

POLICY_PATH   = os.getenv("POLICY_PATH",   str(Path(__file__).parent.parent / "policy_terms.json"))
DATABASE_URL  = os.getenv("DATABASE_URL",  "sqlite:///./data/claims.db")
UPLOAD_DIR    = os.getenv("UPLOAD_DIR",    str(Path(__file__).parent.parent / "uploads"))
OCR_SHOW_LOG  = os.getenv("OCR_SHOW_LOG",  "false").lower() == "true"

# Minimum OCR confidence below which a document is flagged UNREADABLE
OCR_READABILITY_THRESHOLD = 0.30

# Confidence penalty applied to the final score per failed agent
AGENT_FAILURE_CONFIDENCE_PENALTY = 0.20

APP_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Policy wrapper
# ---------------------------------------------------------------------------

@dataclass
class PolicyConfig:
    """
    Wraps the raw policy JSON and exposes typed, named accessors.
    Agents must use these methods rather than indexing the raw dict directly,
    so that a policy schema change only requires a fix here.
    """
    _raw: dict[str, Any] = field(repr=False)

    # ── Identity ────────────────────────────────────────────────────────────

    @property
    def policy_id(self) -> str:
        return self._raw["policy_id"]

    @property
    def policy_start_date(self) -> str:
        return self._raw["policy_holder"]["policy_start_date"]

    @property
    def policy_end_date(self) -> str:
        return self._raw["policy_holder"]["policy_end_date"]

    # ── Members ─────────────────────────────────────────────────────────────

    def get_member(self, member_id: str) -> dict[str, Any] | None:
        """Return the member dict or None if not found."""
        for m in self._raw.get("members", []):
            if m["member_id"] == member_id:
                return m
        return None

    def get_all_members(self) -> list[dict[str, Any]]:
        return self._raw.get("members", [])

    # ── Coverage ─────────────────────────────────────────────────────────────

    @property
    def sum_insured(self) -> float:
        return float(self._raw["coverage"]["sum_insured_per_employee"])

    @property
    def annual_opd_limit(self) -> float:
        return float(self._raw["coverage"]["annual_opd_limit"])

    @property
    def per_claim_limit(self) -> float:
        return float(self._raw["coverage"]["per_claim_limit"])

    # ── OPD category rules ───────────────────────────────────────────────────

    def get_category_rules(self, category: str) -> dict[str, Any] | None:
        """
        Returns the rules dict for a claim category.
        category should be one of: consultation, diagnostic, pharmacy,
        dental, vision, alternative_medicine  (case-insensitive).
        """
        key = category.lower()
        return self._raw.get("opd_categories", {}).get(key)

    def get_sub_limit(self, category: str) -> float | None:
        rules = self.get_category_rules(category)
        if rules and "sub_limit" in rules:
            return float(rules["sub_limit"])
        return None

    def get_copay_percent(self, category: str) -> float:
        rules = self.get_category_rules(category)
        if rules:
            return float(rules.get("copay_percent", 0))
        return 0.0

    def get_network_discount_percent(self, category: str) -> float:
        rules = self.get_category_rules(category)
        if rules:
            return float(rules.get("network_discount_percent", 0))
        return 0.0

    def requires_prescription(self, category: str) -> bool:
        rules = self.get_category_rules(category)
        return bool(rules and rules.get("requires_prescription", False))

    def requires_pre_auth(self, category: str) -> bool:
        rules = self.get_category_rules(category)
        return bool(rules and rules.get("requires_pre_auth", False))

    def pre_auth_threshold(self, category: str) -> float | None:
        rules = self.get_category_rules(category)
        if rules and "pre_auth_threshold" in rules:
            return float(rules["pre_auth_threshold"])
        return None

    def high_value_tests_requiring_pre_auth(self, category: str) -> list[str]:
        rules = self.get_category_rules(category)
        if rules:
            return rules.get("high_value_tests_requiring_pre_auth", [])
        return []

    # ── Dental / Vision inclusion lists ─────────────────────────────────────

    def get_covered_procedures(self, category: str) -> list[str]:
        rules = self.get_category_rules(category)
        if rules:
            return rules.get("covered_procedures", rules.get("covered_items", []))
        return []

    def get_excluded_procedures(self, category: str) -> list[str]:
        rules = self.get_category_rules(category)
        if rules:
            return rules.get("excluded_procedures", rules.get("excluded_items", []))
        return []

    # ── Waiting periods ──────────────────────────────────────────────────────

    @property
    def initial_waiting_period_days(self) -> int:
        return int(self._raw["waiting_periods"]["initial_waiting_period_days"])

    @property
    def pre_existing_waiting_period_days(self) -> int:
        return int(self._raw["waiting_periods"]["pre_existing_conditions_days"])

    def get_condition_waiting_period_days(self, condition_keyword: str) -> int | None:
        """
        Returns waiting period in days for a specific condition keyword,
        or None if the condition has no special waiting period.
        condition_keyword is matched case-insensitively against the keys
        of waiting_periods.specific_conditions.
        """
        specifics: dict[str, int] = self._raw["waiting_periods"].get("specific_conditions", {})
        key = condition_keyword.lower()
        # Direct match first
        if key in specifics:
            return int(specifics[key])
        # Partial match — e.g. "type 2 diabetes mellitus" should hit "diabetes"
        for condition, days in specifics.items():
            if condition in key or key in condition:
                return int(days)
        return None

    # ── Exclusions ───────────────────────────────────────────────────────────

    def get_global_exclusions(self) -> list[str]:
        return self._raw.get("exclusions", {}).get("conditions", [])

    def get_dental_exclusions(self) -> list[str]:
        return self._raw.get("exclusions", {}).get("dental_exclusions", [])

    def get_vision_exclusions(self) -> list[str]:
        return self._raw.get("exclusions", {}).get("vision_exclusions", [])

    def is_excluded(self, text: str) -> tuple[bool, str]:
        """
        Returns (True, matching_exclusion) if the text matches any global
        exclusion. Match is case-insensitive substring.
        """
        text_lower = text.lower()
        for exclusion in self.get_global_exclusions():
            keywords = exclusion.lower().split()
            # Match if all significant words of the exclusion appear in text
            significant = [w for w in keywords if len(w) > 3]
            if significant and all(w in text_lower for w in significant):
                return True, exclusion
        return False, ""

    # ── Pre-authorisation ────────────────────────────────────────────────────

    def get_pre_auth_requirements(self) -> list[str]:
        return self._raw.get("pre_authorization", {}).get("required_for", [])

    # ── Network hospitals ────────────────────────────────────────────────────

    def is_network_hospital(self, hospital_name: str | None) -> bool:
        if not hospital_name:
            return False
        name_lower = hospital_name.lower()
        for net in self._raw.get("network_hospitals", []):
            if net.lower() in name_lower or name_lower in net.lower():
                return True
        return False

    # ── Document requirements ────────────────────────────────────────────────

    def get_required_documents(self, category: str) -> list[str]:
        reqs = self._raw.get("document_requirements", {})
        return reqs.get(category.upper(), {}).get("required", [])

    def get_optional_documents(self, category: str) -> list[str]:
        reqs = self._raw.get("document_requirements", {})
        return reqs.get(category.upper(), {}).get("optional", [])

    # ── Submission rules ─────────────────────────────────────────────────────

    @property
    def submission_deadline_days(self) -> int:
        return int(self._raw["submission_rules"]["deadline_days_from_treatment"])

    @property
    def minimum_claim_amount(self) -> float:
        return float(self._raw["submission_rules"]["minimum_claim_amount"])

    # ── Fraud thresholds ─────────────────────────────────────────────────────

    @property
    def fraud_same_day_limit(self) -> int:
        return int(self._raw["fraud_thresholds"]["same_day_claims_limit"])

    @property
    def fraud_monthly_limit(self) -> int:
        return int(self._raw["fraud_thresholds"]["monthly_claims_limit"])

    @property
    def fraud_high_value_threshold(self) -> float:
        return float(self._raw["fraud_thresholds"]["high_value_claim_threshold"])

    @property
    def fraud_auto_manual_review_above(self) -> float:
        return float(self._raw["fraud_thresholds"]["auto_manual_review_above"])

    @property
    def fraud_score_threshold(self) -> float:
        return float(self._raw["fraud_thresholds"]["fraud_score_manual_review_threshold"])

    # ── Raw access (escape hatch) ────────────────────────────────────────────

    def raw(self) -> dict[str, Any]:
        """Direct access to the full policy dict. Use sparingly."""
        return self._raw


# ---------------------------------------------------------------------------
# Module-level singleton — imported by agents
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_policy() -> PolicyConfig:
    """
    Loads and caches the policy JSON. Called once at startup.
    lru_cache ensures only one PolicyConfig instance exists.
    """
    path = Path(POLICY_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Policy file not found at {path}. "
            "Set the POLICY_PATH environment variable to the correct path."
        )
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return PolicyConfig(_raw=raw)


# Convenience alias used throughout the codebase
policy: PolicyConfig = None  # type: ignore
# Resolved at first import via the getter below so tests can swap it out.


def get_policy() -> PolicyConfig:
    """FastAPI dependency + direct import accessor."""
    return load_policy()
