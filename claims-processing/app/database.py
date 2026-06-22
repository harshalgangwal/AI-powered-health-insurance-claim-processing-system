"""
database.py
-----------
SQLite setup via SQLAlchemy.

Exports:
    engine          — SQLAlchemy engine (used by create_all at startup)
    SessionLocal    — session factory (used in request handlers)
    get_db()        — FastAPI dependency that yields a session and closes it
    init_db()       — called once at app startup to create tables
    save_claim()    — persist a ClaimResult + documents to SQLite
    get_claim()     — retrieve a ClaimResult by claim_id
    list_claims()   — paginated list of ClaimListItem rows
    count_claims_today()   — used by FraudDetector
    count_claims_month()   — used by FraudDetector
"""

from __future__ import annotations

import json
import os
from collections.abc import Generator
from datetime import date, datetime
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.config import DATABASE_URL
from app.models import Base, ClaimListItem, ClaimRecord, ClaimResult, DocumentRecord

# ---------------------------------------------------------------------------
# Engine and session factory
# ---------------------------------------------------------------------------

# check_same_thread=False is required for SQLite when using FastAPI's
# async request handling (multiple threads may access the same connection).
connect_args: dict[str, Any] = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    # SQLite performance: keep a connection pool of 1 (default for SQLite)
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def init_db() -> None:
    """
    Create all tables if they don't exist.
    Safe to call on every startup — SQLAlchemy uses CREATE TABLE IF NOT EXISTS.
    Also ensures the upload directory exists.
    """
    # Ensure data directory exists (SQLite creates the file, but not the dir)
    db_path = DATABASE_URL.replace("sqlite:///", "").replace("sqlite://", "")
    if db_path and db_path != ":memory:":
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

    Base.metadata.create_all(bind=engine)

    upload_dir = os.getenv("UPLOAD_DIR", "./uploads")
    os.makedirs(upload_dir, exist_ok=True)


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------

def get_db() -> Generator[Session, None, None]:
    """
    Yields a SQLAlchemy session for one request, then closes it.

    Usage in a route:
        @app.post("/claims/submit")
        def submit(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def save_claim(
    db: Session,
    result: ClaimResult,
    policy_id: str,
    treatment_date: str | None,
    file_records: list[dict[str, Any]] | None = None,
) -> ClaimRecord:
    """
    Persist a ClaimResult to the claims table.
    Optionally persist associated document metadata to the documents table.

    Args:
        db:             Active SQLAlchemy session.
        result:         The assembled ClaimResult from the orchestrator.
        policy_id:      Policy ID string from the submission.
        treatment_date: ISO date string for the treatment.
        file_records:   List of dicts with keys: claim_id, file_id,
                        original_name, stored_path, document_type, file_size_bytes.

    Returns:
        The persisted ClaimRecord ORM instance.
    """
    record = ClaimRecord(
        claim_id        = result.claim_id,
        member_id       = result.member_id,
        policy_id       = policy_id,
        category        = result.category.value,
        claimed_amount  = result.claimed_amount,
        approved_amount = result.approved_amount,
        decision        = result.decision.value,
        confidence_score = result.confidence_score,
        treatment_date  = treatment_date,
        result_json     = result.model_dump_json(),
        processed_at    = result.processed_at or datetime.utcnow(),
    )
    db.add(record)

    if file_records:
        for fr in file_records:
            doc = DocumentRecord(
                claim_id       = fr["claim_id"],
                file_id        = fr["file_id"],
                original_name  = fr["original_name"],
                stored_path    = fr["stored_path"],
                document_type  = fr.get("document_type"),
                file_size_bytes = fr.get("file_size_bytes"),
            )
            db.add(doc)

    db.commit()
    db.refresh(record)
    return record


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------

def get_claim(db: Session, claim_id: str) -> ClaimResult | None:
    """
    Retrieve the full ClaimResult (with trace) for a given claim_id.
    Returns None if the claim_id is not found.
    """
    record = db.query(ClaimRecord).filter(ClaimRecord.claim_id == claim_id).first()
    if record is None:
        return None
    return record.to_full_result()


def get_claim_record(db: Session, claim_id: str) -> ClaimRecord | None:
    """Return the raw ORM record (useful for partial updates)."""
    return db.query(ClaimRecord).filter(ClaimRecord.claim_id == claim_id).first()


def list_claims(
    db: Session,
    member_id: str | None = None,
    decision: str | None = None,
    skip: int = 0,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """
    Return a paginated list of claim summaries.
    Optionally filter by member_id or decision.
    """
    q = db.query(ClaimRecord)
    if member_id:
        q = q.filter(ClaimRecord.member_id == member_id)
    if decision:
        q = q.filter(ClaimRecord.decision == decision.upper())
    q = q.order_by(ClaimRecord.submitted_at.desc()).offset(skip).limit(limit)
    return [r.to_list_item() for r in q.all()]


# ---------------------------------------------------------------------------
# Fraud detection helpers
# ---------------------------------------------------------------------------

def count_claims_today(db: Session, member_id: str, treatment_date: str) -> int:
    """
    Count claims already processed for this member on treatment_date.
    treatment_date is an ISO date string: "2024-10-30".
    """
    result = db.execute(
        text(
            "SELECT COUNT(*) FROM claims "
            "WHERE member_id = :mid AND treatment_date = :dt"
        ),
        {"mid": member_id, "dt": treatment_date},
    )
    return result.scalar() or 0


def count_claims_this_month(db: Session, member_id: str, treatment_date: str) -> int:
    """
    Count claims processed for this member in the same calendar month as
    treatment_date.
    """
    try:
        d = date.fromisoformat(treatment_date)
        month_start = f"{d.year}-{d.month:02d}-01"
        # Last day of month: use next month minus 1 day
        if d.month == 12:
            month_end = f"{d.year + 1}-01-01"
        else:
            month_end = f"{d.year}-{d.month + 1:02d}-01"
    except (ValueError, AttributeError):
        return 0

    result = db.execute(
        text(
            "SELECT COUNT(*) FROM claims "
            "WHERE member_id = :mid "
            "AND treatment_date >= :start AND treatment_date < :end"
        ),
        {"mid": member_id, "start": month_start, "end": month_end},
    )
    return result.scalar() or 0


def get_ytd_approved_amount(db: Session, member_id: str, policy_year_start: str) -> float:
    """
    Sum of approved amounts for a member since the policy year start date.
    Used by PolicyEngine to check the annual OPD limit.
    """
    result = db.execute(
        text(
            "SELECT COALESCE(SUM(approved_amount), 0) FROM claims "
            "WHERE member_id = :mid "
            "AND treatment_date >= :start "
            "AND decision IN ('APPROVED', 'PARTIAL')"
        ),
        {"mid": member_id, "start": policy_year_start},
    )
    return float(result.scalar() or 0.0)
