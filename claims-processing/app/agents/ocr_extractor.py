"""
ocr_extractor.py
----------------
Agent 2 — OCR Extractor

Responsibilities:
  1. Convert uploaded files (images and PDFs) to images
  2. Run PaddleOCR on each image to get raw text
  3. Apply regex/pattern matching to extract structured fields:
       patient_name, doctor_name, doctor_reg_no, hospital_name,
       diagnosis, treatment_date, line_items, total_amount
  4. Assign per-field confidence; mark low-confidence fields explicitly
  5. Consolidate extracted fields across all documents in a claim
  6. Handle all failures gracefully — never crash the pipeline

No LLMs. All extraction is regex + heuristic pattern matching.

Public interface:
    extractor = OCRExtractor()
    result: ExtractionResult = extractor.extract(files, classifications)

    files          = {file_id: bytes_content}
    classifications = {file_id: DocumentType}
"""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path
from typing import Any

from app.config import OCR_SHOW_LOG, get_policy
from app.models import (
    AgentStatus,
    DocumentType,
    ExtractedDocument,
    ExtractionResult,
    LineItem,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PaddleOCR singleton — initialised once to avoid reloading model weights
# ---------------------------------------------------------------------------

_paddle_ocr_instance = None


def _get_ocr():
    """
    Lazy-initialise PaddleOCR. Returns None if PaddleOCR is unavailable
    (so the rest of the pipeline can degrade gracefully in test environments).
    """
    global _paddle_ocr_instance
    if _paddle_ocr_instance is not None:
        return _paddle_ocr_instance
    try:
        from paddleocr import PaddleOCR  # noqa: PLC0415
        _paddle_ocr_instance = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            show_log=OCR_SHOW_LOG,
            # Disable GPU — use CPU only for portability
            use_gpu=False,
        )
        logger.info("PaddleOCR initialised successfully")
        return _paddle_ocr_instance
    except Exception as exc:
        logger.error("PaddleOCR failed to initialise: %s", exc)
        return None


# ---------------------------------------------------------------------------
# PDF → image conversion
# ---------------------------------------------------------------------------

def pdf_to_images(content: bytes) -> list[Any]:
    """
    Convert PDF bytes to a list of PIL Images, one per page.
    Returns an empty list on failure.
    """
    try:
        from pdf2image import convert_from_bytes  # noqa: PLC0415
        images = convert_from_bytes(content, dpi=200)
        return images
    except Exception as exc:
        logger.warning("pdf2image conversion failed: %s", exc)
        return []


def bytes_to_image(content: bytes) -> Any | None:
    """
    Load image bytes into a PIL Image. Returns None on failure.
    """
    try:
        from PIL import Image  # noqa: PLC0415
        img = Image.open(io.BytesIO(content))
        img = img.convert("RGB")  # normalise mode (handles RGBA, palette, etc.)
        return img
    except Exception as exc:
        logger.warning("Image load failed: %s", exc)
        return None


def image_to_numpy(pil_image: Any) -> Any | None:
    """Convert PIL Image to numpy array for PaddleOCR."""
    try:
        import numpy as np  # noqa: PLC0415
        return np.array(pil_image)
    except Exception as exc:
        logger.warning("numpy conversion failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Raw OCR
# ---------------------------------------------------------------------------

def run_ocr(content: bytes, filename: str) -> tuple[str, list, float]:
    """
    Run PaddleOCR on a file's bytes.

    Returns:
        (raw_text, paddle_lines, avg_confidence)
        raw_text      — all recognised text joined by newlines
        paddle_lines  — raw PaddleOCR output (list of line results)
        avg_confidence — mean confidence across all detected text lines
    """
    ext = Path(filename).suffix.lower()
    ocr = _get_ocr()

    # Determine images to process
    images: list[Any] = []
    if ext == ".pdf":
        images = pdf_to_images(content)
        if not images:
            logger.warning("PDF conversion yielded no images for %s", filename)
            return "", [], 0.0
    else:
        img = bytes_to_image(content)
        if img is None:
            return "", [], 0.0
        images = [img]

    all_lines: list = []
    all_text_parts: list[str] = []
    all_confidences: list[float] = []

    for page_img in images:
        if ocr is None:
            # PaddleOCR unavailable — return empty so pipeline degrades
            logger.warning("PaddleOCR unavailable; skipping OCR for %s", filename)
            break

        numpy_img = image_to_numpy(page_img)
        if numpy_img is None:
            continue

        try:
            result = ocr.ocr(numpy_img, cls=True)
        except Exception as exc:
            logger.warning("PaddleOCR.ocr() raised: %s", exc)
            continue

        if not result:
            continue

        # result is a list of pages; each page is a list of lines
        for page in result:
            if not page:
                continue
            for line in page:
                try:
                    text = line[1][0]
                    conf = float(line[1][1])
                    all_lines.append(line)
                    all_text_parts.append(text)
                    all_confidences.append(conf)
                except (IndexError, TypeError, ValueError):
                    continue

    raw_text   = "\n".join(all_text_parts)
    avg_conf   = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
    return raw_text, all_lines, avg_conf


# ---------------------------------------------------------------------------
# Regex patterns for field extraction
# ---------------------------------------------------------------------------

# ── Dates ────────────────────────────────────────────────────────────────────
# Matches DD-MM-YYYY, DD/MM/YYYY, DD.MM.YYYY, and some variants
_DATE_PATTERNS = [
    re.compile(r"\b(\d{2}[-/.]\d{2}[-/.]\d{4})\b"),        # DD-MM-YYYY
    re.compile(r"\b(\d{4}[-/.]\d{2}[-/.]\d{2})\b"),         # YYYY-MM-DD
    re.compile(r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\b", re.IGNORECASE),
    re.compile(r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b", re.IGNORECASE),
    re.compile(r"\bdate\s*[:\-]?\s*(\d{2}[-/.]\d{2}[-/.]\d{4})\b", re.IGNORECASE),
    re.compile(r"\bdate\s*[:\-]?\s*(\d{1,2}[-/.]\w{3}[-/.]\d{4})\b", re.IGNORECASE),
]

# ── Patient name ─────────────────────────────────────────────────────────────
_PATIENT_NAME_PATTERNS = [
    re.compile(r"patient\s+name\s*[:\-]\s*([A-Za-z][A-Za-z\s\.]{2,40})", re.IGNORECASE),
    re.compile(r"patient\s*[:\-]\s*([A-Za-z][A-Za-z\s\.]{2,40})", re.IGNORECASE),
    re.compile(r"name\s*[:\-]\s*([A-Za-z][A-Za-z\s\.]{2,40})", re.IGNORECASE),
    re.compile(r"(?:mr|mrs|ms|master|miss)\.?\s+([A-Za-z][A-Za-z\s\.]{2,35})", re.IGNORECASE),
]

# ── Doctor name ───────────────────────────────────────────────────────────────
_DOCTOR_NAME_PATTERNS = [
    re.compile(r"(?:dr\.?|doctor)\s+([A-Za-z][A-Za-z\s\.]{2,40})", re.IGNORECASE),
    re.compile(r"ref(?:erring)?\s+doctor\s*[:\-]\s*(?:dr\.?)?\s*([A-Za-z][A-Za-z\s\.]{2,40})", re.IGNORECASE),
    re.compile(r"physician\s*[:\-]\s*(?:dr\.?)?\s*([A-Za-z][A-Za-z\s\.]{2,40})", re.IGNORECASE),
    re.compile(r"consultant\s*[:\-]\s*(?:dr\.?)?\s*([A-Za-z][A-Za-z\s\.]{2,40})", re.IGNORECASE),
]

# ── Doctor registration number ────────────────────────────────────────────────
# Formats from sample_documents_guide.md:
# KA/45678/2015, MH/23456/2018, AYUR/KL/2345/2019 etc.
_REG_NO_PATTERNS = [
    re.compile(r"\b(AYUR/[A-Z]{2}/\d{4,6}/\d{4})\b", re.IGNORECASE),
    re.compile(r"\b([A-Z]{2}/\d{4,6}/\d{4})\b", re.IGNORECASE),
    re.compile(r"reg(?:istration)?\.?\s*no\.?\s*[:\-]?\s*([A-Z0-9/\-]{6,20})", re.IGNORECASE),
    re.compile(r"reg\.?\s*#?\s*[:\-]?\s*([A-Z]{2}/\d{4,6}/\d{4})", re.IGNORECASE),
]

# ── Diagnosis ─────────────────────────────────────────────────────────────────
_DIAGNOSIS_PATTERNS = [
    re.compile(r"diagnosis\s*[:\-]\s*(.+?)(?:\n|$)", re.IGNORECASE),
    re.compile(r"provisional\s+diagnosis\s*[:\-]\s*(.+?)(?:\n|$)", re.IGNORECASE),
    re.compile(r"final\s+diagnosis\s*[:\-]\s*(.+?)(?:\n|$)", re.IGNORECASE),
    re.compile(r"dx\s*[:\-]\s*(.+?)(?:\n|$)", re.IGNORECASE),
    re.compile(r"impression\s*[:\-]\s*(.+?)(?:\n|$)", re.IGNORECASE),
]

# ── Hospital / clinic name ────────────────────────────────────────────────────
_HOSPITAL_PATTERNS = [
    re.compile(r"(?:hospital|clinic|centre|center|medical|health)\s+name\s*[:\-]\s*(.+?)(?:\n|$)", re.IGNORECASE),
    # First line of a bill is often the facility name — caught by positional logic below
]

# ── Total amount ──────────────────────────────────────────────────────────────
_TOTAL_PATTERNS = [
    re.compile(r"(?:grand\s+)?total\s+(?:amount\s+)?(?:paid\s+)?[:\-]?\s*(?:rs\.?|₹|inr)?\s*([\d,]+(?:\.\d{1,2})?)", re.IGNORECASE),
    re.compile(r"net\s+(?:amount\s+)?(?:payable\s+)?[:\-]?\s*(?:rs\.?|₹|inr)?\s*([\d,]+(?:\.\d{1,2})?)", re.IGNORECASE),
    re.compile(r"amount\s+(?:paid\s+)?[:\-]?\s*(?:rs\.?|₹|inr)?\s*([\d,]+(?:\.\d{1,2})?)", re.IGNORECASE),
    re.compile(r"(?:rs\.?|₹|inr)\s*([\d,]+(?:\.\d{1,2})?)\s*(?:/-|only|total)?", re.IGNORECASE),
]

# ── Line items ────────────────────────────────────────────────────────────────
# Matches lines like:  "Consultation Fee    1    1000.00    1000.00"
# or                   "Root Canal Treatment              8000"
_LINE_ITEM_PATTERN = re.compile(
    r"^(.{5,60?}?)\s{2,}(?:\d+\s+)?(?:[\d,]+\.?\d*\s+)?([\d,]+\.\d{2})\s*$",
    re.MULTILINE,
)
# Simpler fallback: "Description   Amount" where amount is at end of line
_LINE_ITEM_SIMPLE = re.compile(
    r"^([\w\s\/\(\)\-\.]{5,50})\s+([\d,]+(?:\.\d{1,2})?)\s*$",
    re.MULTILINE,
)

# Words that disqualify a "line item" description (header/footer noise)
_LINE_ITEM_STOPWORDS = {
    "total", "subtotal", "sub total", "grand total", "net amount", "balance",
    "discount", "gst", "tax", "cgst", "sgst", "igst", "paid", "received",
    "description", "amount", "qty", "rate", "item", "particulars",
    "bill no", "date", "patient", "doctor",
}


# ---------------------------------------------------------------------------
# Individual field extractors
# ---------------------------------------------------------------------------

def _first_match(patterns: list[re.Pattern], text: str) -> str | None:
    """Return the first captured group from the first matching pattern."""
    for p in patterns:
        m = p.search(text)
        if m:
            raw = m.group(1).strip()
            # Trim to the first newline in case the capture ran long
            raw = raw.split("\n")[0].strip()
            if raw:
                return raw
    return None


def _clean_amount(raw: str) -> float | None:
    """Parse '1,500.00' or '1500' into a float. Returns None on failure."""
    try:
        return float(raw.replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def extract_date(text: str) -> str | None:
    """Extract the first recognisable date string and normalise to YYYY-MM-DD."""
    for p in _DATE_PATTERNS:
        m = p.search(text)
        if m:
            raw = m.group(1).strip()
            return _normalise_date(raw)
    return None


_MONTH_MAP = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "may": "05", "jun": "06", "jul": "07", "aug": "08",
    "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}


def _normalise_date(raw: str) -> str:
    """
    Convert various date strings to ISO format YYYY-MM-DD.
    Returns the raw string unchanged if parsing fails.
    """
    raw = raw.strip()
    # YYYY-MM-DD already
    if re.match(r"^\d{4}[-/]\d{2}[-/]\d{2}$", raw):
        return raw.replace("/", "-")

    # DD-MM-YYYY or DD/MM/YYYY or DD.MM.YYYY
    m = re.match(r"^(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})$", raw)
    if m:
        d, mo, y = m.group(1), m.group(2), m.group(3)
        return f"{y}-{mo.zfill(2)}-{d.zfill(2)}"

    # DD Mon YYYY or Mon DD, YYYY
    m2 = re.match(r"^(\d{1,2})\s+([A-Za-z]{3})[a-z]*\s+(\d{4})$", raw)
    if m2:
        d, mon, y = m2.group(1), m2.group(2).lower()[:3], m2.group(3)
        mo = _MONTH_MAP.get(mon, "01")
        return f"{y}-{mo}-{d.zfill(2)}"

    m3 = re.match(r"^([A-Za-z]{3})[a-z]*\s+(\d{1,2}),?\s+(\d{4})$", raw)
    if m3:
        mon, d, y = m3.group(1).lower()[:3], m3.group(2), m3.group(3)
        mo = _MONTH_MAP.get(mon, "01")
        return f"{y}-{mo}-{d.zfill(2)}"

    return raw  # Return as-is; caller marks as low confidence


def extract_patient_name(text: str) -> str | None:
    raw = _first_match(_PATIENT_NAME_PATTERNS, text)
    if not raw:
        return None
    # Trim at first digit or common noise tokens
    raw = re.split(r"[\d\|/\\]", raw)[0].strip()
    raw = re.sub(r"\s+", " ", raw)
    return raw.title() if len(raw) >= 3 else None


def extract_doctor_name(text: str) -> str | None:
    raw = _first_match(_DOCTOR_NAME_PATTERNS, text)
    if not raw:
        return None
    raw = re.split(r"[\d\|/\\,]", raw)[0].strip()
    raw = re.sub(r"\s+", " ", raw)
    # Strip trailing qualification noise (MBBS, MD, etc.)
    raw = re.sub(r"\b(mbbs|md|ms|dnb|dgo|frcs|mch|phd|bds)\b.*$", "", raw, flags=re.IGNORECASE).strip()
    return raw.title() if len(raw) >= 3 else None


def extract_doctor_reg_no(text: str) -> str | None:
    return _first_match(_REG_NO_PATTERNS, text)


def extract_diagnosis(text: str) -> str | None:
    raw = _first_match(_DIAGNOSIS_PATTERNS, text)
    if not raw:
        return None
    # Clean up common artifacts
    raw = re.sub(r"[|\\]", "", raw).strip()
    raw = re.sub(r"\s+", " ", raw)
    return raw if len(raw) >= 3 else None


def extract_hospital_name(text: str, doc_type: DocumentType) -> str | None:
    """
    For bills/invoices the facility name is usually on the first 1-2 lines.
    For prescriptions it appears after the doctor name.
    """
    # Try explicit label first
    explicit = _first_match(_HOSPITAL_PATTERNS, text)
    if explicit:
        return explicit.strip().title()

    # For bills: take the first non-empty line that looks like a facility name
    if doc_type in (DocumentType.HOSPITAL_BILL, DocumentType.PHARMACY_BILL):
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        for line in lines[:4]:
            # Skip lines that are clearly not a name (contain only numbers, dates, etc.)
            if re.search(r"(bill|receipt|invoice|gstin|date|patient)", line, re.IGNORECASE):
                continue
            if re.search(r"(hospital|clinic|centre|center|medical|pharmacy|health)", line, re.IGNORECASE):
                return line.title()
    return None


def extract_total_amount(text: str) -> float | None:
    """Return the highest unambiguous total amount found in the text."""
    candidates: list[float] = []
    for p in _TOTAL_PATTERNS:
        for m in p.finditer(text):
            val = _clean_amount(m.group(1))
            if val is not None and val > 0:
                candidates.append(val)
    if not candidates:
        return None
    # Return the maximum — total is typically the largest amount on a bill
    return max(candidates)


def extract_line_items(text: str) -> list[LineItem]:
    """
    Extract itemised line items from bill/invoice text.
    Uses two regex patterns; deduplicates by description.
    """
    items: list[LineItem] = []
    seen_descriptions: set[str] = set()

    def _add(description: str, amount_str: str) -> None:
        desc = description.strip()
        desc = re.sub(r"\s+", " ", desc)
        # Skip header/footer noise
        if desc.lower() in _LINE_ITEM_STOPWORDS:
            return
        if any(sw in desc.lower() for sw in _LINE_ITEM_STOPWORDS):
            # Allow partial matches only if description is substantive
            if len(desc.split()) <= 2:
                return
        val = _clean_amount(amount_str)
        if val is None or val <= 0:
            return
        key = desc.lower()
        if key in seen_descriptions:
            return
        seen_descriptions.add(key)
        items.append(LineItem(description=desc, amount=val))

    # Try detailed pattern first
    for m in _LINE_ITEM_PATTERN.finditer(text):
        _add(m.group(1), m.group(2))

    # Fallback to simple pattern
    if not items:
        for m in _LINE_ITEM_SIMPLE.finditer(text):
            _add(m.group(1), m.group(2))

    return items


# ---------------------------------------------------------------------------
# Per-document extractor
# ---------------------------------------------------------------------------

def extract_document_fields(
    raw_text: str,
    file_id:  str,
    doc_type: DocumentType,
    ocr_confidence: float,
) -> ExtractedDocument:
    """
    Run all field extractors against raw_text and build an ExtractedDocument.
    Marks fields as low_confidence if they could not be extracted or if overall
    OCR confidence is below 0.60.
    """
    low_conf_fields: list[str]  = []
    flags:           list[str]  = []

    # ── Extract each field ────────────────────────────────────────────────────
    patient_name   = extract_patient_name(raw_text)
    doctor_name    = extract_doctor_name(raw_text)
    doctor_reg_no  = extract_doctor_reg_no(raw_text)
    hospital_name  = extract_hospital_name(raw_text, doc_type)
    diagnosis      = extract_diagnosis(raw_text)
    treatment_date = extract_date(raw_text)
    total_amount   = extract_total_amount(raw_text)

    # Line items only for bill types
    line_items: list[LineItem] = []
    if doc_type in (DocumentType.HOSPITAL_BILL, DocumentType.PHARMACY_BILL, DocumentType.DENTAL_REPORT):
        line_items = extract_line_items(raw_text)

    # ── Mark low-confidence fields ────────────────────────────────────────────
    if ocr_confidence < 0.60:
        # Flag every extracted field as potentially inaccurate
        low_conf_fields.append("all_fields_low_ocr")

    if patient_name is None:
        low_conf_fields.append("patient_name")
    if doctor_name is None and doc_type == DocumentType.PRESCRIPTION:
        low_conf_fields.append("doctor_name")
    if diagnosis is None and doc_type == DocumentType.PRESCRIPTION:
        low_conf_fields.append("diagnosis")
    if total_amount is None and doc_type in (
        DocumentType.HOSPITAL_BILL, DocumentType.PHARMACY_BILL
    ):
        low_conf_fields.append("total_amount")
    if treatment_date is None:
        low_conf_fields.append("treatment_date")

    # ── Fraud-relevant flags ──────────────────────────────────────────────────
    # Detect crossed-out / altered amounts (common fraud signal)
    if re.search(r"(?:corrected|cancelled|revised|amended)", raw_text, re.IGNORECASE):
        flags.append("DOCUMENT_ALTERATION")
    # Detect duplicate stamps
    if raw_text.lower().count("original") > 1 or raw_text.lower().count("duplicate") > 1:
        flags.append("DUPLICATE_STAMP")

    # ── Normalise doctor name prefix ──────────────────────────────────────────
    if doctor_name and not doctor_name.lower().startswith("dr"):
        doctor_name = f"Dr. {doctor_name}"

    return ExtractedDocument(
        file_id              = file_id,
        document_type        = doc_type,
        patient_name         = patient_name,
        doctor_name          = doctor_name,
        doctor_reg_no        = doctor_reg_no,
        hospital_name        = hospital_name,
        diagnosis            = diagnosis,
        treatment_date       = treatment_date,
        line_items           = line_items,
        total_amount         = total_amount,
        ocr_confidence       = round(ocr_confidence, 3),
        low_confidence_fields = low_conf_fields,
        flags                = flags,
    )


# ---------------------------------------------------------------------------
# Consolidator — merge fields across multiple documents
# ---------------------------------------------------------------------------

def consolidate(docs: list[ExtractedDocument]) -> dict:
    """
    Merge extracted fields from multiple documents into a single set of
    consolidated values for the PolicyEngine to use.

    Priority rules:
    - patient_name:    from PRESCRIPTION > HOSPITAL_BILL > others
    - diagnosis:       from PRESCRIPTION only
    - hospital_name:   from HOSPITAL_BILL > others
    - total_amount:    from HOSPITAL_BILL (sum of all bills)
    - treatment_date:  earliest date found
    """
    def _pick(field: str, priority_types: list[DocumentType]) -> Any:
        # Try priority types first
        for pt in priority_types:
            for d in docs:
                if d.document_type == pt:
                    val = getattr(d, field)
                    if val:
                        return val
        # Fall back to any document
        for d in docs:
            val = getattr(d, field)
            if val:
                return val
        return None

    patient_name   = _pick("patient_name",  [DocumentType.PRESCRIPTION, DocumentType.HOSPITAL_BILL])
    doctor_name    = _pick("doctor_name",   [DocumentType.PRESCRIPTION])
    diagnosis      = _pick("diagnosis",     [DocumentType.PRESCRIPTION])
    hospital_name  = _pick("hospital_name", [DocumentType.HOSPITAL_BILL])

    # Total: sum all bill amounts; if none found, take max of any total
    bill_totals = [
        d.total_amount for d in docs
        if d.document_type in (DocumentType.HOSPITAL_BILL, DocumentType.PHARMACY_BILL)
        and d.total_amount is not None
    ]
    total_amount: float | None = sum(bill_totals) if bill_totals else _pick("total_amount", [])

    # Treatment date: take the earliest valid date across all docs
    dates = [d.treatment_date for d in docs if d.treatment_date]
    treatment_date = min(dates) if dates else None

    # Overall confidence: minimum across all documents (weakest link)
    confs = [d.ocr_confidence for d in docs]
    overall_confidence = min(confs) if confs else 0.0

    return {
        "patient_name":        patient_name,
        "doctor_name":         doctor_name,
        "diagnosis":           diagnosis,
        "hospital_name":       hospital_name,
        "total_amount":        total_amount,
        "treatment_date":      treatment_date,
        "overall_confidence":  overall_confidence,
    }


# ---------------------------------------------------------------------------
# Main extractor class
# ---------------------------------------------------------------------------

class OCRExtractor:
    """
    Agent 2 — OCR Extractor.

    Usage:
        extractor = OCRExtractor()
        result = extractor.extract(
            files           = {"file_001": b"...bytes..."},
            file_names      = {"file_001": "prescription.jpg"},
            classifications = {"file_001": DocumentType.PRESCRIPTION},
        )
    """

    def __init__(self) -> None:
        self._policy = get_policy()

    def extract(
        self,
        files:           dict[str, bytes],        # file_id → raw bytes
        file_names:      dict[str, str],           # file_id → original filename
        classifications: dict[str, DocumentType],  # file_id → classified type
    ) -> ExtractionResult:
        """
        Run OCR and field extraction on all files.

        Args:
            files:           Raw file bytes keyed by file_id.
            file_names:      Original filenames keyed by file_id.
            classifications: Document types from Agent 1, keyed by file_id.

        Returns:
            ExtractionResult with per-document and consolidated fields.
        """
        try:
            return self._run_extraction(files, file_names, classifications)
        except Exception as exc:
            logger.exception("OCRExtractor crashed: %s", exc)
            return ExtractionResult(
                documents          = [],
                overall_confidence = 0.0,
                agent_status       = AgentStatus.FAILED,
                agent_error        = str(exc),
            )

    def _run_extraction(
        self,
        files:           dict[str, bytes],
        file_names:      dict[str, str],
        classifications: dict[str, DocumentType],
    ) -> ExtractionResult:

        extracted_docs: list[ExtractedDocument] = []

        for file_id, content in files.items():
            fname    = file_names.get(file_id, file_id)
            doc_type = classifications.get(file_id, DocumentType.UNKNOWN)

            try:
                raw_text, _paddle_lines, ocr_conf = run_ocr(content, fname)

                if not raw_text.strip():
                    logger.warning("No text extracted from %s", fname)
                    # Still produce a record with low confidence
                    extracted_docs.append(ExtractedDocument(
                        file_id               = file_id,
                        document_type         = doc_type,
                        ocr_confidence        = 0.0,
                        low_confidence_fields = ["all_fields_ocr_failed"],
                        flags                 = ["OCR_NO_TEXT"],
                    ))
                    continue

                doc = extract_document_fields(raw_text, file_id, doc_type, ocr_conf)
                extracted_docs.append(doc)

            except Exception as exc:
                logger.warning("Extraction failed for %s: %s", fname, exc)
                extracted_docs.append(ExtractedDocument(
                    file_id               = file_id,
                    document_type         = doc_type,
                    ocr_confidence        = 0.0,
                    low_confidence_fields = ["all_fields_extraction_error"],
                    flags                 = [f"EXTRACTION_ERROR: {str(exc)[:80]}"],
                ))

        # Consolidate fields across all successfully extracted documents
        consolidated = consolidate(extracted_docs)

        return ExtractionResult(
            documents          = extracted_docs,
            patient_name       = consolidated["patient_name"],
            hospital_name      = consolidated["hospital_name"],
            diagnosis          = consolidated["diagnosis"],
            total_amount       = consolidated["total_amount"],
            treatment_date     = consolidated["treatment_date"],
            overall_confidence = round(consolidated["overall_confidence"], 3),
            agent_status       = AgentStatus.SUCCESS,
        )
