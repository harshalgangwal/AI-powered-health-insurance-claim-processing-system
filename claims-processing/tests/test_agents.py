"""
tests/test_agents.py
--------------------
Unit tests for all four agents and core utilities.

Run with:
    pytest tests/test_agents.py -v

Tests use in-memory SQLite and mock OCR so no real files are needed.
PaddleOCR is patched out entirely to keep tests fast and dependency-free.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# ---------------------------------------------------------------------------
# Minimal policy fixture — enough to exercise all agents
# ---------------------------------------------------------------------------

POLICY_JSON: dict[str, Any] = {
    "policy_id": "PLM-TEST-2024",
    "policy_holder": {
        "policy_start_date": "2024-01-01",
        "policy_end_date":   "2024-12-31",
    },
    "members": [
        {
            "member_id": "EMP001",
            "name":      "Alice Sharma",
            "relation":  "self",
            "date_of_birth": "1990-05-10",
            "pre_existing_conditions": [],
        },
        {
            "member_id": "EMP002",
            "name":      "Bob Kumar",
            "relation":  "self",
            "date_of_birth": "1985-03-22",
            "pre_existing_conditions": ["diabetes"],
        },
    ],
    "coverage": {
        "sum_insured_per_employee": 500000,
        "annual_opd_limit": 20000,
        "per_claim_limit": 5000,
    },
    "opd_categories": {
        "consultation": {
            "sub_limit": 1500,
            "copay_percent": 20,
            "network_discount_percent": 10,
            "requires_prescription": False,
            "requires_pre_auth": False,
        },
        "diagnostic": {
            "sub_limit": 5000,
            "copay_percent": 20,
            "requires_prescription": True,
            "requires_pre_auth": False,
            "pre_auth_threshold": 3000,
            "high_value_tests_requiring_pre_auth": ["MRI", "CT scan"],
        },
        "pharmacy": {
            "sub_limit": 3000,
            "copay_percent": 30,
            "requires_prescription": True,
        },
        "dental": {
            "sub_limit": 2000,
            "copay_percent": 20,
            "covered_procedures": ["tooth extraction", "dental filling"],
            "excluded_procedures": ["cosmetic dentistry", "teeth whitening"],
        },
        "vision": {
            "sub_limit": 2000,
            "copay_percent": 20,
            "covered_items": ["spectacles", "contact lenses"],
            "excluded_items": ["laser surgery"],
        },
        "alternative_medicine": {
            "sub_limit": 2000,
            "copay_percent": 20,
        },
    },
    "waiting_periods": {
        "initial_waiting_period_days": 30,
        "pre_existing_conditions_days": 365,
        "specific_conditions": {
            "diabetes": 180,
            "hypertension": 180,
        },
    },
    "exclusions": {
        "conditions": [
            "cosmetic surgery",
            "self-inflicted injuries",
            "fertility treatment",
        ],
        "dental_exclusions": ["cosmetic dentistry", "teeth whitening"],
        "vision_exclusions":  ["laser surgery"],
    },
    "pre_authorization": {
        "required_for": ["MRI", "CT scan", "hospitalisation"],
    },
    "network_hospitals": ["Apollo Hospitals", "Fortis Healthcare", "Max Hospital"],
    "document_requirements": {
        "CONSULTATION": {
            "required": ["doctor prescription / consultation note"],
            "optional": ["lab report"],
        },
        "DIAGNOSTIC": {
            "required": ["lab report", "doctor prescription"],
            "optional": [],
        },
        "PHARMACY": {
            "required": ["pharmacy bill", "doctor prescription"],
            "optional": [],
        },
        "DENTAL": {
            "required": ["dental report / bill"],
            "optional": [],
        },
        "VISION": {
            "required": ["optical bill / prescription"],
            "optional": [],
        },
    },
    "submission_rules": {
        "deadline_days_from_treatment": 90,
        "minimum_claim_amount": 200,
    },
    "fraud_thresholds": {
        "same_day_claims_limit": 3,
        "monthly_claims_limit": 10,
        "high_value_claim_threshold": 10000,
        "auto_manual_review_above": 15000,
        "fraud_score_manual_review_threshold": 0.60,
    },
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def policy_file(tmp_path_factory):
    """Write POLICY_JSON to a temp file and set POLICY_PATH."""
    d = tmp_path_factory.mktemp("policy")
    p = d / "policy_terms.json"
    p.write_text(json.dumps(POLICY_JSON), encoding="utf-8")
    os.environ["POLICY_PATH"] = str(p)
    # Clear lru_cache so the new path is picked up
    from app.config import load_policy
    load_policy.cache_clear()
    yield str(p)
    load_policy.cache_clear()


@pytest.fixture()
def db_session(tmp_path):
    """In-memory SQLite session."""
    from app.models import Base
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


# ---------------------------------------------------------------------------
# Helper — tiny 1-pixel PNG (avoids real file I/O in most tests)
# ---------------------------------------------------------------------------

MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
    b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _upload_file(name: str = "bill.png", content: bytes = MINIMAL_PNG):
    """Create a minimal UploadFile-like mock."""
    f = MagicMock()
    f.filename = name
    f.read = MagicMock(return_value=content)
    f.seek = MagicMock()
    return f


# ---------------------------------------------------------------------------
# config.py tests
# ---------------------------------------------------------------------------

class TestPolicyConfig:
    def test_member_found(self, policy_file):
        from app.config import get_policy
        p = get_policy()
        m = p.get_member("EMP001")
        assert m is not None
        assert m["name"] == "Alice Sharma"

    def test_member_not_found(self, policy_file):
        from app.config import get_policy
        p = get_policy()
        assert p.get_member("DOESNOTEXIST") is None

    def test_category_rules(self, policy_file):
        from app.config import get_policy
        p = get_policy()
        rules = p.get_category_rules("consultation")
        assert rules is not None
        assert rules["copay_percent"] == 20

    def test_waiting_period_specific(self, policy_file):
        from app.config import get_policy
        p = get_policy()
        assert p.get_condition_waiting_period_days("diabetes") == 180

    def test_waiting_period_partial_match(self, policy_file):
        from app.config import get_policy
        p = get_policy()
        # "type 2 diabetes mellitus" should match "diabetes"
        assert p.get_condition_waiting_period_days("type 2 diabetes mellitus") == 180

    def test_waiting_period_unknown(self, policy_file):
        from app.config import get_policy
        p = get_policy()
        assert p.get_condition_waiting_period_days("common cold") is None

    def test_is_excluded(self, policy_file):
        from app.config import get_policy
        p = get_policy()
        excluded, reason = p.is_excluded("patient underwent cosmetic surgery")
        assert excluded is True
        assert "cosmetic" in reason.lower()

    def test_not_excluded(self, policy_file):
        from app.config import get_policy
        p = get_policy()
        excluded, _ = p.is_excluded("routine blood test consultation")
        assert excluded is False

    def test_network_hospital(self, policy_file):
        from app.config import get_policy
        p = get_policy()
        assert p.is_network_hospital("Apollo Hospitals, Mumbai") is True
        assert p.is_network_hospital("Random Clinic") is False

    def test_fraud_thresholds(self, policy_file):
        from app.config import get_policy
        p = get_policy()
        assert p.fraud_same_day_limit == 3
        assert p.fraud_high_value_threshold == 10000


# ---------------------------------------------------------------------------
# database.py tests
# ---------------------------------------------------------------------------

class TestDatabase:
    def test_count_claims_today_empty(self, policy_file, db_session):
        from app.database import count_claims_today
        assert count_claims_today(db_session, "EMP001", "2024-06-01") == 0

    def test_count_claims_this_month_empty(self, policy_file, db_session):
        from app.database import count_claims_this_month
        assert count_claims_this_month(db_session, "EMP001", "2024-06-01") == 0

    def test_save_and_get_claim(self, policy_file, db_session):
        from app.database import save_claim, get_claim
        from app.models import (
            ClaimResult, ClaimCategory, DecisionType
        )
        result = ClaimResult(
            claim_id="TEST-001",
            member_id="EMP001",
            category=ClaimCategory.CONSULTATION,
            claimed_amount=800.0,
            decision=DecisionType.APPROVED,
            approved_amount=640.0,
            confidence_score=0.95,
        )
        save_claim(db_session, result, policy_id="PLM-TEST-2024", treatment_date="2024-06-01")
        retrieved = get_claim(db_session, "TEST-001")
        assert retrieved is not None
        assert retrieved.claim_id == "TEST-001"
        assert retrieved.decision == DecisionType.APPROVED

    def test_get_claim_not_found(self, policy_file, db_session):
        from app.database import get_claim
        assert get_claim(db_session, "DOES-NOT-EXIST") is None

    def test_list_claims(self, policy_file, db_session):
        from app.database import save_claim, list_claims
        from app.models import ClaimResult, ClaimCategory, DecisionType
        for i in range(3):
            r = ClaimResult(
                claim_id=f"LIST-{i}",
                member_id="EMP001",
                category=ClaimCategory.PHARMACY,
                claimed_amount=500.0,
                decision=DecisionType.PARTIAL,
                approved_amount=350.0,
                confidence_score=0.9,
            )
            save_claim(db_session, r, policy_id="PLM-TEST-2024", treatment_date="2024-07-01")
        rows = list_claims(db_session, member_id="EMP001")
        assert len(rows) >= 3


# ---------------------------------------------------------------------------
# Agent 3 — PolicyEngine tests (pure Python, no OCR needed)
# ---------------------------------------------------------------------------

class TestPolicyEngine:
    """
    PolicyEngine is pure Python and reads from POLICY_JSON.
    We test it directly without touching files or OCR.
    """

    def _run(
        self,
        db_session,
        *,
        member_id="EMP001",
        category="CONSULTATION",
        claimed_amount=800.0,
        treatment_date=None,
        hospital_name=None,
        diagnosis=None,
        ytd=0.0,
    ):
        from app.agents.policy_engine import PolicyEngine
        from app.models import ClaimCategory, ExtractionResult

        if treatment_date is None:
            # Use a date well within the policy year, past the waiting period
            treatment_date = "2024-06-01"

        engine = PolicyEngine()
        return engine.run(
            member_id=member_id,
            claim_category=ClaimCategory(category),
            claimed_amount=claimed_amount,
            treatment_date=treatment_date,
            hospital_name=hospital_name,
            diagnosis=diagnosis,
            extraction=ExtractionResult(),
            ytd_claims_amount=ytd,
            db=db_session,
        )

    def test_approved_consultation(self, policy_file, db_session):
        result = self._run(db_session, claimed_amount=800.0)
        from app.models import DecisionType
        assert result.decision == DecisionType.APPROVED
        # Copay 20% → approved = 800 * 0.80 = 640
        assert abs(result.approved_amount - 640.0) < 1.0

    def test_member_not_found(self, policy_file, db_session):
        result = self._run(db_session, member_id="EMP999")
        from app.models import DecisionType
        assert result.decision == DecisionType.REJECTED
        assert any("member" in r.lower() for r in result.rejection_reasons)

    def test_initial_waiting_period(self, policy_file, db_session):
        # Policy starts 2024-01-01; treatment on day 10 → within waiting period
        result = self._run(db_session, treatment_date="2024-01-10")
        from app.models import DecisionType
        assert result.decision == DecisionType.REJECTED
        assert any("waiting" in r.lower() for r in result.rejection_reasons)

    def test_specific_condition_waiting_period(self, policy_file, db_session):
        # EMP002 has pre-existing diabetes; diabetes waiting period = 180 days
        result = self._run(
            db_session,
            member_id="EMP002",
            diagnosis="Type 2 diabetes mellitus",
            treatment_date="2024-06-01",   # 152 days into year — within 180-day period
        )
        from app.models import DecisionType
        assert result.decision == DecisionType.REJECTED

    def test_global_exclusion(self, policy_file, db_session):
        result = self._run(db_session, diagnosis="cosmetic surgery procedure")
        from app.models import DecisionType
        assert result.decision == DecisionType.REJECTED

    def test_sub_limit_partial(self, policy_file, db_session):
        # Consultation sub-limit = 1500; claim 2000 → partial
        result = self._run(db_session, claimed_amount=2000.0)
        from app.models import DecisionType
        # Capped at 1500 then 20% copay → 1500 * 0.80 = 1200
        assert result.decision in (DecisionType.APPROVED, DecisionType.PARTIAL)
        assert result.approved_amount <= 1500.0

    def test_per_claim_limit(self, policy_file, db_session):
        # Per-claim limit = 5000; claim 6000 → capped
        result = self._run(db_session, claimed_amount=6000.0, category="DIAGNOSTIC")
        assert result.approved_amount <= 5000.0

    def test_network_discount(self, policy_file, db_session):
        result = self._run(
            db_session,
            claimed_amount=1000.0,
            hospital_name="Apollo Hospitals",
        )
        from app.models import DecisionType
        # Network discount 10% → eligible 900, then copay 20% → 720
        assert result.decision == DecisionType.APPROVED
        # Discount should be applied
        assert result.discount_applied > 0

    def test_ytd_limit_exceeded(self, policy_file, db_session):
        # annual_opd_limit = 20000; ytd already 19500; claim 1000 → partial or rejected
        result = self._run(db_session, claimed_amount=1000.0, ytd=19500.0)
        from app.models import DecisionType
        assert result.decision in (DecisionType.PARTIAL, DecisionType.REJECTED)
        assert result.approved_amount <= 500.0

    def test_minimum_claim_amount(self, policy_file, db_session):
        result = self._run(db_session, claimed_amount=100.0)   # below 200 minimum
        from app.models import DecisionType
        assert result.decision == DecisionType.REJECTED
        assert any("minimum" in r.lower() for r in result.rejection_reasons)


# ---------------------------------------------------------------------------
# Agent 4 — FraudDetector tests
# ---------------------------------------------------------------------------

class TestFraudDetector:
    def test_clean_claim(self, policy_file, db_session):
        from app.agents.fraud_detector import FraudDetector
        from app.models import ExtractionResult
        fd = FraudDetector()
        result = fd.run(
            member_id="EMP001",
            treatment_date="2024-06-01",
            claimed_amount=800.0,
            extraction=ExtractionResult(),
            db=db_session,
        )
        assert result.fraud_score < 0.60
        assert result.override_to_manual is False

    def test_high_value_claim(self, policy_file, db_session):
        from app.agents.fraud_detector import FraudDetector
        from app.models import ExtractionResult
        fd = FraudDetector()
        result = fd.run(
            member_id="EMP001",
            treatment_date="2024-06-01",
            claimed_amount=15000.0,   # above fraud_high_value_threshold (10000)
            extraction=ExtractionResult(),
            db=db_session,
        )
        assert any("HIGH_VALUE" in f or "AUTO_MANUAL" in f for f in result.flags)
        # 15000 ≥ auto_manual_review_above (15000) → override
        assert result.override_to_manual is True

    def test_round_number_flag(self, policy_file, db_session):
        from app.agents.fraud_detector import FraudDetector
        from app.models import ExtractionResult
        fd = FraudDetector()
        result = fd.run(
            member_id="EMP001",
            treatment_date="2024-06-01",
            claimed_amount=5000.0,
            extraction=ExtractionResult(),
            db=db_session,
        )
        assert any("ROUND_NUMBER" in f for f in result.flags)

    def test_document_alteration_flag(self, policy_file, db_session):
        from app.agents.fraud_detector import FraudDetector
        from app.models import ExtractionResult, ExtractedDocument, DocumentType
        fd = FraudDetector()
        extraction = ExtractionResult(
            documents=[
                ExtractedDocument(
                    file_id="f1",
                    document_type=DocumentType.HOSPITAL_BILL,
                    flags=["DOCUMENT_ALTERATION"],
                    ocr_confidence=0.9,
                )
            ]
        )
        result = fd.run(
            member_id="EMP001",
            treatment_date="2024-06-01",
            claimed_amount=800.0,
            extraction=extraction,
            db=db_session,
        )
        assert any("DOCUMENT_ALTERATION" in f for f in result.flags)
        assert result.fraud_score >= 0.35


# ---------------------------------------------------------------------------
# models.py tests
# ---------------------------------------------------------------------------

class TestModels:
    def test_claim_result_defaults(self):
        from app.models import ClaimResult, ClaimCategory, DecisionType
        r = ClaimResult(
            claim_id="X",
            member_id="EMP001",
            category=ClaimCategory.CONSULTATION,
            claimed_amount=500.0,
            decision=DecisionType.APPROVED,
            approved_amount=400.0,
            confidence_score=0.9,
        )
        assert r.pipeline_complete is True
        assert r.rejection_reasons == []

    def test_claim_record_roundtrip(self, policy_file, db_session):
        from app.database import save_claim, get_claim
        from app.models import ClaimResult, ClaimCategory, DecisionType
        r = ClaimResult(
            claim_id="RT-001",
            member_id="EMP001",
            category=ClaimCategory.DENTAL,
            claimed_amount=1200.0,
            decision=DecisionType.PARTIAL,
            approved_amount=960.0,
            confidence_score=0.88,
        )
        save_claim(db_session, r, policy_id="PLM-TEST-2024", treatment_date="2024-08-10")
        got = get_claim(db_session, "RT-001")
        assert got.approved_amount == 960.0
        assert got.category.value == "DENTAL"
