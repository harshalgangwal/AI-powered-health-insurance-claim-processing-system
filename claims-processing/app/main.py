"""
main.py
-------
FastAPI application entry point.

Routes:
    GET  /              → serves static/index.html
    GET  /health        → HealthResponse
    POST /claims/submit → run the 4-agent pipeline, return ClaimResult
    GET  /claims/       → paginated list of processed claims
    GET  /claims/{id}   → full ClaimResult with trace

Orchestrator:
    _process_claim() runs Agents 1-4 in sequence.
    Each agent call is wrapped in try/except so one failure never crashes
    the pipeline — the agent is marked FAILED and processing continues.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import aiofiles
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from app.config import APP_VERSION, UPLOAD_DIR, get_policy
from app.database import (
    get_db,
    get_claim,
    init_db,
    list_claims,
    save_claim,
)
from app.models import (
    AgentStatus,
    ClaimCategory,
    ClaimListItem,
    ClaimResult,
    ClaimSubmitRequest,
    DecisionType,
    ErrorDetail,
    ExtractionResult,
    FraudResult,
    HealthResponse,
    PolicyDecision,
    DocVerificationResult,
)
from app.agents.doc_verifier import DocVerifier
from app.agents.ocr_extractor import OCRExtractor
from app.agents.policy_engine import PolicyEngine
from app.agents.fraud_detector import FraudDetector

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Health Insurance Claims Processing",
    version=APP_VERSION,
    description="Automated OPD claims adjudication — document verification → OCR → policy → fraud",
)

# Serve static files (index.html UI)
_static_dir = Path(__file__).parent.parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    # Eagerly load policy so any config error surfaces at startup, not per-request
    get_policy()
    logger.info("Claims Processing API v%s started.", APP_VERSION)


# ---------------------------------------------------------------------------
# Agent singletons (initialised once at startup)
# ---------------------------------------------------------------------------

_doc_verifier = DocVerifier()
_ocr_extractor = OCRExtractor()
_fraud_detector = FraudDetector()


def _get_policy_engine() -> PolicyEngine:
    """PolicyEngine is lightweight — create per-request so it picks up a fresh DB session."""
    return PolicyEngine()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONFIDENCE_PENALTY = 0.20   # per failed agent


def _penalty(score: float, n: int = 1) -> float:
    return max(0.0, round(score - CONFIDENCE_PENALTY * n, 4))


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

async def _process_claim(
    request: ClaimSubmitRequest,
    files: list[UploadFile],
    db: Session,
) -> ClaimResult:
    """
    Run the 4-agent pipeline for a single claim submission.

    Pipeline:
        Agent 1 (DocVerifier)   — classify & validate documents
        Agent 2 (OCRExtractor)  — extract structured data from files
        Agent 3 (PolicyEngine)  — adjudicate against policy rules
        Agent 4 (FraudDetector) — score for fraud; may escalate to MANUAL_REVIEW

    Every agent is wrapped in try/except. A crashed agent sets its status to
    FAILED, reduces confidence, and the pipeline continues with the next agent.
    """
    claim_id = str(uuid.uuid4())
    pipeline_notes: list[str] = []
    confidence = 1.0
    failed_agents = 0

    # ── Persist uploaded files to disk ──────────────────────────────────────
    claim_upload_dir = Path(UPLOAD_DIR) / claim_id
    claim_upload_dir.mkdir(parents=True, exist_ok=True)

    file_records: list[dict[str, Any]] = []
    saved_paths: list[Path] = []

    for f in files:
        file_id = str(uuid.uuid4())
        dest = claim_upload_dir / f.filename
        content = await f.read()
        async with aiofiles.open(dest, "wb") as out:
            await out.write(content)
        saved_paths.append(dest)
        file_records.append(
            {
                "claim_id": claim_id,
                "file_id": file_id,
                "original_name": f.filename,
                "stored_path": str(dest),
                "file_size_bytes": len(content),
            }
        )
        # Reset so agents can read the file again if needed
        await f.seek(0)

    # ── Agent 1: Document Verifier ──────────────────────────────────────────
    doc_verification: DocVerificationResult | None = None
    try:
        doc_verification = _doc_verifier.run(
            files=files,
            saved_paths=saved_paths,
            claim_category=request.claim_category,
        )
        # Update file_records with classified document types
        for fr, cls in zip(file_records, doc_verification.classifications):
            fr["document_type"] = cls.classified_as.value

        if not doc_verification.passed:
            # Hard stop — return immediately with REJECTED
            logger.info("claim=%s Agent1 hard-stop: %s", claim_id, doc_verification.errors)
            result = ClaimResult(
                claim_id=claim_id,
                member_id=request.member_id,
                category=request.claim_category,
                claimed_amount=request.claimed_amount,
                decision=DecisionType.REJECTED,
                approved_amount=0.0,
                rejection_reasons=doc_verification.errors,
                confidence_score=round(confidence, 4),
                doc_verification=doc_verification,
                pipeline_complete=False,
                pipeline_notes=["Pipeline stopped after Agent 1: document check failed."],
                processed_at=datetime.utcnow(),
            )
            save_claim(
                db, result,
                policy_id=request.policy_id,
                treatment_date=request.treatment_date,
                file_records=file_records,
            )
            return result

    except Exception as exc:
        logger.exception("claim=%s Agent1 crashed: %s", claim_id, exc)
        doc_verification = DocVerificationResult(
            passed=True,   # allow pipeline to continue
            errors=[],
            agent_status=AgentStatus.FAILED,
            agent_error=str(exc),
        )
        confidence = _penalty(confidence)
        failed_agents += 1
        pipeline_notes.append(f"Agent 1 (DocVerifier) failed: {exc}")

    # ── Agent 2: OCR Extractor ──────────────────────────────────────────────
    extraction: ExtractionResult | None = None
    try:
        extraction = _ocr_extractor.run(
            files=files,
            saved_paths=saved_paths,
            classifications=doc_verification.classifications if doc_verification else [],
        )
        if extraction.agent_status == AgentStatus.FAILED:
            confidence = _penalty(confidence)
            failed_agents += 1
            pipeline_notes.append(f"Agent 2 (OCRExtractor) degraded: {extraction.agent_error}")

    except Exception as exc:
        logger.exception("claim=%s Agent2 crashed: %s", claim_id, exc)
        extraction = ExtractionResult(
            agent_status=AgentStatus.FAILED,
            agent_error=str(exc),
        )
        confidence = _penalty(confidence)
        failed_agents += 1
        pipeline_notes.append(f"Agent 2 (OCRExtractor) failed: {exc}")

    # Merge extraction values with explicit submission fields
    # (Submitted values take precedence for financial fields)
    effective_amount = request.claimed_amount
    effective_hospital = request.hospital_name or (extraction.hospital_name if extraction else None)
    effective_diagnosis = extraction.diagnosis if extraction else None
    effective_treatment_date = request.treatment_date

    # ── Agent 3: Policy Engine ──────────────────────────────────────────────
    policy_decision: PolicyDecision | None = None
    try:
        engine = _get_policy_engine()
        policy_decision = engine.run(
            member_id=request.member_id,
            claim_category=request.claim_category,
            claimed_amount=effective_amount,
            treatment_date=effective_treatment_date,
            hospital_name=effective_hospital,
            diagnosis=effective_diagnosis,
            extraction=extraction,
            ytd_claims_amount=request.ytd_claims_amount,
            db=db,
        )
        if policy_decision.agent_status == AgentStatus.FAILED:
            confidence = _penalty(confidence)
            failed_agents += 1
            pipeline_notes.append(f"Agent 3 (PolicyEngine) degraded: {policy_decision.agent_error}")

    except Exception as exc:
        logger.exception("claim=%s Agent3 crashed: %s", claim_id, exc)
        policy_decision = PolicyDecision(
            decision=DecisionType.MANUAL_REVIEW,
            claimed_amount=effective_amount,
            agent_status=AgentStatus.FAILED,
            agent_error=str(exc),
        )
        confidence = _penalty(confidence)
        failed_agents += 1
        pipeline_notes.append(f"Agent 3 (PolicyEngine) failed: {exc}")

    # ── Agent 4: Fraud Detector ─────────────────────────────────────────────
    fraud_result: FraudResult | None = None
    try:
        fraud_result = _fraud_detector.run(
            member_id=request.member_id,
            treatment_date=effective_treatment_date,
            claimed_amount=effective_amount,
            extraction=extraction,
            db=db,
        )
        if fraud_result.agent_status == AgentStatus.FAILED:
            confidence = _penalty(confidence)
            failed_agents += 1
            pipeline_notes.append(f"Agent 4 (FraudDetector) degraded: {fraud_result.agent_error}")

    except Exception as exc:
        logger.exception("claim=%s Agent4 crashed: %s", claim_id, exc)
        fraud_result = FraudResult(
            agent_status=AgentStatus.FAILED,
            agent_error=str(exc),
        )
        confidence = _penalty(confidence)
        failed_agents += 1
        pipeline_notes.append(f"Agent 4 (FraudDetector) failed: {exc}")

    # ── Assemble final decision ─────────────────────────────────────────────
    final_decision = policy_decision.decision if policy_decision else DecisionType.MANUAL_REVIEW
    approved_amount = policy_decision.approved_amount if policy_decision else 0.0
    rejection_reasons = policy_decision.rejection_reasons if policy_decision else []

    # Fraud override: escalate to MANUAL_REVIEW (never downgrade an approved)
    if fraud_result and fraud_result.override_to_manual:
        if final_decision == DecisionType.APPROVED:
            final_decision = DecisionType.MANUAL_REVIEW
            pipeline_notes.append("Decision escalated to MANUAL_REVIEW by fraud detector.")

    # Incorporate OCR confidence into overall score
    if extraction and extraction.overall_confidence < 1.0:
        confidence = min(confidence, extraction.overall_confidence + 0.10)

    # Incorporate policy confidence
    if policy_decision and policy_decision.confidence < 1.0:
        confidence = min(confidence, policy_decision.confidence)

    pipeline_complete = failed_agents == 0

    result = ClaimResult(
        claim_id=claim_id,
        member_id=request.member_id,
        category=request.claim_category,
        claimed_amount=effective_amount,
        decision=final_decision,
        approved_amount=approved_amount,
        rejection_reasons=rejection_reasons,
        confidence_score=round(confidence, 4),
        doc_verification=doc_verification,
        extraction=extraction,
        policy_decision=policy_decision,
        fraud_result=fraud_result,
        pipeline_complete=pipeline_complete,
        pipeline_notes=pipeline_notes,
        processed_at=datetime.utcnow(),
    )

    save_claim(
        db, result,
        policy_id=request.policy_id,
        treatment_date=effective_treatment_date,
        file_records=file_records,
    )

    logger.info(
        "claim=%s member=%s category=%s amount=%.0f decision=%s confidence=%.2f",
        claim_id,
        request.member_id,
        request.claim_category.value,
        effective_amount,
        final_decision.value,
        confidence,
    )
    return result


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def serve_index() -> FileResponse:
    index = Path(__file__).parent.parent / "static" / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(str(index))


@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=APP_VERSION)


@app.post(
    "/claims/submit",
    response_model=ClaimResult,
    tags=["Claims"],
    summary="Submit an OPD claim for adjudication",
    responses={
        400: {"model": ErrorDetail, "description": "Validation error"},
        422: {"description": "Missing required fields"},
    },
)
async def submit_claim(
    # Form fields
    member_id: str = Form(...),
    policy_id: str = Form(...),
    claim_category: ClaimCategory = Form(...),
    treatment_date: str = Form(...),
    claimed_amount: float = Form(...),
    ytd_claims_amount: float = Form(0.0),
    simulate_component_failure: bool = Form(False),
    hospital_name: str | None = Form(None),
    # Files
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
) -> ClaimResult:
    """
    Accept a multipart/form-data claim submission.

    Required form fields: member_id, policy_id, claim_category,
    treatment_date (ISO: YYYY-MM-DD), claimed_amount.

    Required files: at least one document (PDF, PNG, JPG, JPEG).
    """
    if not files or all(f.filename == "" for f in files):
        raise HTTPException(
            status_code=400,
            detail="At least one document file is required.",
        )

    # Validate file extensions
    allowed = {".pdf", ".png", ".jpg", ".jpeg"}
    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext not in allowed:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{ext}'. Allowed: PDF, PNG, JPG, JPEG.",
            )

    request = ClaimSubmitRequest(
        member_id=member_id,
        policy_id=policy_id,
        claim_category=claim_category,
        treatment_date=treatment_date,
        claimed_amount=claimed_amount,
        ytd_claims_amount=ytd_claims_amount,
        simulate_component_failure=simulate_component_failure,
        hospital_name=hospital_name,
    )

    return await _process_claim(request, files, db)


@app.get(
    "/claims/",
    response_model=list[ClaimListItem],
    tags=["Claims"],
    summary="List processed claims (paginated)",
)
def get_claims(
    member_id: str | None = None,
    decision: str | None = None,
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
) -> list[dict]:
    return list_claims(db, member_id=member_id, decision=decision, skip=skip, limit=limit)


@app.get(
    "/claims/{claim_id}",
    response_model=ClaimResult,
    tags=["Claims"],
    summary="Get full claim result with agent trace",
    responses={404: {"model": ErrorDetail}},
)
def get_claim_by_id(
    claim_id: str,
    db: Session = Depends(get_db),
) -> ClaimResult:
    result = get_claim(db, claim_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Claim '{claim_id}' not found.")
    return result
