"""
models.py
---------
Single source of truth for all data shapes in the system.

Section 1 — Enums                   shared constants
Section 2 — Agent output schemas    what each agent returns
Section 3 — API schemas             request/response for HTTP layer
Section 4 — SQLAlchemy ORM          what gets persisted to SQLite
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase


# ---------------------------------------------------------------------------
# Section 1 — Enums
# ---------------------------------------------------------------------------

class ClaimCategory(str, Enum):
    CONSULTATION        = "CONSULTATION"
    DIAGNOSTIC          = "DIAGNOSTIC"
    PHARMACY            = "PHARMACY"
    DENTAL              = "DENTAL"
    VISION              = "VISION"
    ALTERNATIVE_MEDICINE = "ALTERNATIVE_MEDICINE"


class DocumentType(str, Enum):
    PRESCRIPTION    = "PRESCRIPTION"
    HOSPITAL_BILL   = "HOSPITAL_BILL"
    LAB_REPORT      = "LAB_REPORT"
    PHARMACY_BILL   = "PHARMACY_BILL"
    DENTAL_REPORT   = "DENTAL_REPORT"
    DISCHARGE_SUMMARY = "DISCHARGE_SUMMARY"
    UNKNOWN         = "UNKNOWN"


class DecisionType(str, Enum):
    APPROVED        = "APPROVED"
    PARTIAL         = "PARTIAL"
    REJECTED        = "REJECTED"
    MANUAL_REVIEW   = "MANUAL_REVIEW"


class AgentStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED  = "FAILED"   # exception caught by orchestrator; pipeline continued
    SKIPPED = "SKIPPED"  # not run because an earlier agent halted the pipeline


class DocumentQuality(str, Enum):
    GOOD        = "GOOD"
    LOW         = "LOW"        # readable but noisy
    UNREADABLE  = "UNREADABLE" # OCR confidence below threshold
    EMPTY       = "EMPTY"      # blank image or zero-byte file


# ---------------------------------------------------------------------------
# Section 2 — Agent output schemas
# ---------------------------------------------------------------------------

# ── Agent 1: Document Verifier ──────────────────────────────────────────────

class DocumentClassification(BaseModel):
    """Result for a single uploaded file after classification."""
    file_id:        str
    file_name:      str
    classified_as:  DocumentType
    quality:        DocumentQuality
    patient_name:   str | None = None   # extracted for cross-doc name check
    confidence:     float = Field(ge=0.0, le=1.0)
    notes:          str | None = None


class DocVerificationResult(BaseModel):
    """
    Output of Agent 1.
    If passed=False, the pipeline stops. errors contains specific messages
    that are forwarded directly to the member.
    """
    passed:           bool
    errors:           list[str] = Field(default_factory=list)
    classifications:  list[DocumentClassification] = Field(default_factory=list)
    agent_status:     AgentStatus = AgentStatus.SUCCESS
    agent_error:      str | None = None   # populated if agent itself crashed


# ── Agent 2: OCR Extractor ──────────────────────────────────────────────────

class LineItem(BaseModel):
    description:    str
    amount:         float
    included:       bool | None = None   # set by PolicyEngine later
    exclusion_reason: str | None = None


class ExtractedDocument(BaseModel):
    """Structured data extracted from one document."""
    file_id:            str
    document_type:      DocumentType
    # Patient / provider
    patient_name:       str | None = None
    doctor_name:        str | None = None
    doctor_reg_no:      str | None = None
    hospital_name:      str | None = None
    # Clinical
    diagnosis:          str | None = None
    diagnosis_codes:    list[str] = Field(default_factory=list)
    medicines:          list[str] = Field(default_factory=list)
    # Financial
    line_items:         list[LineItem] = Field(default_factory=list)
    total_amount:       float | None = None
    # Dates
    treatment_date:     str | None = None   # ISO date string
    # Confidence
    ocr_confidence:     float = Field(default=1.0, ge=0.0, le=1.0)
    low_confidence_fields: list[str] = Field(default_factory=list)
    flags:              list[str] = Field(default_factory=list)  # e.g. DOCUMENT_ALTERATION


class ExtractionResult(BaseModel):
    """Output of Agent 2."""
    documents:          list[ExtractedDocument] = Field(default_factory=list)
    # Merged/consolidated values across all documents in this claim
    patient_name:       str | None = None
    hospital_name:      str | None = None
    diagnosis:          str | None = None
    total_amount:       float | None = None
    treatment_date:     str | None = None
    overall_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    agent_status:       AgentStatus = AgentStatus.SUCCESS
    agent_error:        str | None = None


# ── Agent 3: Policy Engine ───────────────────────────────────────────────────

class CheckResult(BaseModel):
    """
    A single policy check, recorded for the trace.
    Every check the engine runs — pass or fail — appears here.
    """
    check_name:     str
    passed:         bool
    detail:         str             # human-readable explanation
    impact:         str | None = None  # e.g. "₹360 co-pay deducted"


class PolicyDecision(BaseModel):
    """Output of Agent 3."""
    decision:           DecisionType
    approved_amount:    float = 0.0
    claimed_amount:     float = 0.0
    copay_applied:      float = 0.0
    discount_applied:   float = 0.0
    rejection_reasons:  list[str] = Field(default_factory=list)
    line_item_decisions: list[LineItem] = Field(default_factory=list)
    checks:             list[CheckResult] = Field(default_factory=list)
    confidence:         float = Field(default=1.0, ge=0.0, le=1.0)
    agent_status:       AgentStatus = AgentStatus.SUCCESS
    agent_error:        str | None = None


# ── Agent 4: Fraud Detector ──────────────────────────────────────────────────

class FraudResult(BaseModel):
    """Output of Agent 4."""
    fraud_score:            float = Field(default=0.0, ge=0.0, le=1.0)
    flags:                  list[str] = Field(default_factory=list)
    override_to_manual:     bool = False
    same_day_claim_count:   int = 0
    monthly_claim_count:    int = 0
    agent_status:           AgentStatus = AgentStatus.SUCCESS
    agent_error:            str | None = None


# ── Assembled result ─────────────────────────────────────────────────────────

class ClaimResult(BaseModel):
    """
    Final output of the orchestrator. Returned by POST /claims/submit
    and stored (as JSON) in the claims table.
    """
    claim_id:           str
    member_id:          str
    category:           ClaimCategory
    claimed_amount:     float
    # Final adjudication
    decision:           DecisionType
    approved_amount:    float = 0.0
    rejection_reasons:  list[str] = Field(default_factory=list)
    # Confidence — degraded when agents fail or fields are low-confidence
    confidence_score:   float = Field(ge=0.0, le=1.0)
    # Full trace for observability
    doc_verification:   DocVerificationResult | None = None
    extraction:         ExtractionResult | None = None
    policy_decision:    PolicyDecision | None = None
    fraud_result:       FraudResult | None = None
    # Pipeline health
    pipeline_complete:  bool = True   # False if any agent was FAILED or SKIPPED
    pipeline_notes:     list[str] = Field(default_factory=list)
    # Timestamps
    submitted_at:       datetime = Field(default_factory=datetime.utcnow)
    processed_at:       datetime | None = None


# ---------------------------------------------------------------------------
# Section 3 — API schemas
# ---------------------------------------------------------------------------

class ClaimSubmitRequest(BaseModel):
    """
    Parsed from the multipart form fields (files handled separately).
    All fields come in as strings from the form; Pydantic coerces them.
    """
    member_id:       str
    policy_id:       str
    claim_category:  ClaimCategory
    treatment_date:  str   # ISO date: "2024-11-01"
    claimed_amount:  float
    # Optional — provided in test cases; real members won't send these
    ytd_claims_amount:        float = 0.0
    simulate_component_failure: bool = False
    hospital_name:   str | None = None


class ClaimListItem(BaseModel):
    """Summary row for GET /claims/ list endpoint."""
    claim_id:       str
    member_id:      str
    category:       str
    claimed_amount: float
    decision:       str
    approved_amount: float
    confidence_score: float
    submitted_at:   datetime


class HealthResponse(BaseModel):
    status:  str = "ok"
    version: str = "1.0.0"


class ErrorDetail(BaseModel):
    """Structured error body returned on 4xx responses."""
    error:   str
    detail:  str | None = None
    # For document errors: what was found vs what is needed
    uploaded_types:  list[str] = Field(default_factory=list)
    required_types:  list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Section 4 — SQLAlchemy ORM (SQLite)
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class ClaimRecord(Base):
    """
    Persists every processed claim.
    The full ClaimResult is stored as a JSON blob in `result_json` so
    the complete trace is always retrievable, even if the schema evolves.
    """
    __tablename__ = "claims"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    claim_id        = Column(String(64), unique=True, nullable=False, index=True)
    member_id       = Column(String(32), nullable=False, index=True)
    policy_id       = Column(String(64), nullable=False)
    category        = Column(String(32), nullable=False)
    claimed_amount  = Column(Float, nullable=False)
    approved_amount = Column(Float, nullable=False, default=0.0)
    decision        = Column(String(16), nullable=False, index=True)
    confidence_score = Column(Float, nullable=False, default=0.0)
    treatment_date  = Column(String(16), nullable=True)
    # Full serialised ClaimResult for trace retrieval
    result_json     = Column(Text, nullable=False)
    submitted_at    = Column(DateTime, server_default=func.now(), nullable=False)
    processed_at    = Column(DateTime, nullable=True)

    def to_list_item(self) -> dict[str, Any]:
        return {
            "claim_id":       self.claim_id,
            "member_id":      self.member_id,
            "category":       self.category,
            "claimed_amount": self.claimed_amount,
            "decision":       self.decision,
            "approved_amount": self.approved_amount,
            "confidence_score": self.confidence_score,
            "submitted_at":   self.submitted_at.isoformat() if self.submitted_at else None,
        }

    def to_full_result(self) -> ClaimResult:
        data = json.loads(self.result_json)
        return ClaimResult(**data)


class DocumentRecord(Base):
    """
    Metadata for each file uploaded with a claim.
    The file is stored on disk at UPLOAD_DIR/{claim_id}/{file_name}.
    """
    __tablename__ = "documents"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    claim_id        = Column(String(64), nullable=False, index=True)
    file_id         = Column(String(64), nullable=False)
    original_name   = Column(String(256), nullable=False)
    stored_path     = Column(String(512), nullable=False)
    document_type   = Column(String(32), nullable=True)   # classified type
    file_size_bytes = Column(Integer, nullable=True)
    uploaded_at     = Column(DateTime, server_default=func.now(), nullable=False)
