"""
doc_verifier.py
---------------
Agent 1 — Document Verifier

Responsibilities:
  1. Classify each uploaded file into a DocumentType using OCR text + keyword rules
  2. Verify the classified types satisfy the required types for the claim category
  3. Flag unreadable documents (OCR confidence below threshold)
  4. Detect patient name mismatches across documents

No LLMs. No vision models. Classification is keyword/heuristic based on OCR output.

Public interface:
    verifier = DocumentVerifier()
    result: DocVerificationResult = verifier.verify(files, category, member_name)

Raises:
    Nothing — all errors are returned inside DocVerificationResult.errors.
    The orchestrator inspects result.passed to decide whether to continue.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from app.config import OCR_READABILITY_THRESHOLD, get_policy
from app.models import (
    AgentStatus,
    DocVerificationResult,
    DocumentClassification,
    DocumentQuality,
    DocumentType,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Keyword rules for document classification
# ---------------------------------------------------------------------------
# Each entry is (DocumentType, required_keywords, optional_boost_keywords).
# A document is classified as the type whose required_keywords all appear
# in the OCR text (case-insensitive). If multiple types match, the one with
# more total keyword hits wins.
#
# Order matters for tie-breaking: more specific types come first.

@dataclass
class ClassificationRule:
    doc_type:  DocumentType
    required:  list[str]          # ALL must be present
    boosters:  list[str] = field(default_factory=list)  # add weight if present


CLASSIFICATION_RULES: list[ClassificationRule] = [
    # ── Pharmacy Bill ────────────────────────────────────────────────────────
    # Check before HOSPITAL_BILL because pharmacy bills also say "bill"
    ClassificationRule(
        doc_type  = DocumentType.PHARMACY_BILL,
        required  = ["pharmacy"],
        boosters  = ["drug", "medicine", "batch", "expiry", "mrp", "pharmacist",
                     "drug lic", "chemist", "dispensed", "tablet", "capsule"],
    ),
    # ── Prescription ─────────────────────────────────────────────────────────
    ClassificationRule(
        doc_type  = DocumentType.PRESCRIPTION,
        required  = ["rx"],
        boosters  = ["diagnosis", "prescription", "doctor", "dr.", "mbbs", "md",
                     "reg. no", "registration", "dosage", "tab ", "cap ", "syp ",
                     "follow-up", "follow up", "chief complaint", "advised"],
    ),
    ClassificationRule(
        doc_type  = DocumentType.PRESCRIPTION,
        required  = ["prescription"],
        boosters  = ["doctor", "dr.", "diagnosis", "medicine", "dosage"],
    ),
    ClassificationRule(
        doc_type  = DocumentType.PRESCRIPTION,
        required  = ["diagnosis", "doctor"],
        boosters  = ["rx", "mbbs", "reg. no", "tab ", "advised", "follow-up"],
    ),
    # ── Lab / Diagnostic Report ───────────────────────────────────────────────
    ClassificationRule(
        doc_type  = DocumentType.LAB_REPORT,
        required  = ["lab"],
        boosters  = ["report", "result", "test", "normal range", "reference range",
                     "sample", "patholog", "nabl", "hemoglobin", "wbc", "cbc",
                     "platelet", "glucose", "creatinine", "urine", "culture",
                     "sensitivity", "pathologist", "specimen"],
    ),
    ClassificationRule(
        doc_type  = DocumentType.LAB_REPORT,
        required  = ["test", "result", "normal range"],
        boosters  = ["patholog", "sample date", "report date", "nabl"],
    ),
    # ── Discharge Summary ────────────────────────────────────────────────────
    ClassificationRule(
        doc_type  = DocumentType.DISCHARGE_SUMMARY,
        required  = ["discharge"],
        boosters  = ["summary", "admission", "discharge date", "ward", "bed",
                     "inpatient", "ip no", "procedure", "operation"],
    ),
    # ── Dental Report ────────────────────────────────────────────────────────
    ClassificationRule(
        doc_type  = DocumentType.DENTAL_REPORT,
        required  = ["dental"],
        boosters  = ["tooth", "teeth", "oral", "root canal", "extraction",
                     "crown", "scaling", "gum", "dentist", "periodont",
                     "orthodont", "caries", "x-ray"],
    ),
    # ── Hospital Bill ─────────────────────────────────────────────────────────
    # Broadest rule — checked last so it doesn't swallow more specific types
    ClassificationRule(
        doc_type  = DocumentType.HOSPITAL_BILL,
        required  = ["bill"],
        boosters  = ["receipt", "invoice", "hospital", "clinic", "total amount",
                     "consultation", "patient", "gstin", "bill no", "amount",
                     "subtotal", "cashier", "payment"],
    ),
    ClassificationRule(
        doc_type  = DocumentType.HOSPITAL_BILL,
        required  = ["invoice"],
        boosters  = ["hospital", "clinic", "total", "patient", "consultation"],
    ),
    ClassificationRule(
        doc_type  = DocumentType.HOSPITAL_BILL,
        required  = ["receipt"],
        boosters  = ["hospital", "clinic", "total", "payment", "patient"],
    ),
]


# ---------------------------------------------------------------------------
# Patient name extraction patterns
# ---------------------------------------------------------------------------
# These regex patterns extract a name from OCR text for cross-document
# patient name matching.

_NAME_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"patient\s*[:\-]?\s*([A-Za-z][A-Za-z\s\.]{2,40})", re.IGNORECASE),
    re.compile(r"patient\s+name\s*[:\-]?\s*([A-Za-z][A-Za-z\s\.]{2,40})", re.IGNORECASE),
    re.compile(r"name\s*[:\-]?\s*([A-Za-z][A-Za-z\s\.]{2,40})", re.IGNORECASE),
    re.compile(r"mr\.?\s+([A-Za-z][A-Za-z\s\.]{2,30})", re.IGNORECASE),
    re.compile(r"mrs\.?\s+([A-Za-z][A-Za-z\s\.]{2,30})", re.IGNORECASE),
    re.compile(r"ms\.?\s+([A-Za-z][A-Za-z\s\.]{2,30})", re.IGNORECASE),
]

# Words that should not be accepted as patient names
_NAME_STOPWORDS: set[str] = {
    "date", "age", "gender", "male", "female", "doctor", "dr", "hospital",
    "clinic", "address", "phone", "mobile", "email", "ref", "referring",
    "report", "bill", "total", "amount", "diagnosis", "prescription",
}


def _extract_name_from_text(text: str) -> str | None:
    """
    Try each name pattern in order; return the first plausible match.
    Returns None if no reliable name found.
    """
    for pattern in _NAME_PATTERNS:
        m = pattern.search(text)
        if m:
            raw = m.group(1).strip()
            # Trim trailing garbage (digits, punctuation after the name)
            raw = re.split(r"[\d\n\r|/\\]", raw)[0].strip()
            raw = re.sub(r"\s+", " ", raw)
            words = raw.lower().split()
            # Reject if any word is a stopword or the name is suspiciously short/long
            if len(words) < 1 or len(words) > 5:
                continue
            if any(w in _NAME_STOPWORDS for w in words):
                continue
            if len(raw) < 3:
                continue
            return raw.title()
    return None


def _names_match(name_a: str | None, name_b: str | None) -> bool:
    """
    Fuzzy name comparison — normalise and check for substring overlap.
    Returns True if the names are likely the same person.
    """
    if not name_a or not name_b:
        # If one name is missing we cannot confirm a mismatch
        return True

    def normalise(n: str) -> set[str]:
        # Lowercase, remove punctuation, split into parts
        n = re.sub(r"[^\w\s]", "", n.lower())
        return {w for w in n.split() if len(w) > 1}

    parts_a = normalise(name_a)
    parts_b = normalise(name_b)

    if not parts_a or not parts_b:
        return True

    # Names match if they share at least one significant word
    overlap = parts_a & parts_b
    return len(overlap) > 0


# ---------------------------------------------------------------------------
# Scorer — picks the best classification for a block of OCR text
# ---------------------------------------------------------------------------

def _score_text(text: str, rule: ClassificationRule) -> int:
    """
    Returns a match score for text against a classification rule.
    0 means the rule does not apply (a required keyword is absent).
    >0 means the rule applies; higher = stronger match.
    """
    text_lower = text.lower()
    # All required keywords must be present
    for kw in rule.required:
        if kw.lower() not in text_lower:
            return 0
    score = len(rule.required) * 10
    for kw in rule.boosters:
        if kw.lower() in text_lower:
            score += 1
    return score


def classify_text(text: str) -> tuple[DocumentType, float]:
    """
    Classify a document based on its OCR text.

    Returns:
        (DocumentType, confidence)
        confidence is in [0, 1] — higher means clearer classification signal.
        Returns (UNKNOWN, 0.0) if no rule matches.
    """
    if not text or not text.strip():
        return DocumentType.UNKNOWN, 0.0

    best_type  = DocumentType.UNKNOWN
    best_score = 0

    for rule in CLASSIFICATION_RULES:
        score = _score_text(text, rule)
        if score > best_score:
            best_score = score
            best_type  = rule.doc_type

    if best_score == 0:
        return DocumentType.UNKNOWN, 0.0

    # Normalise confidence: cap at ~20 points → 1.0
    confidence = min(best_score / 20.0, 1.0)
    return best_type, confidence


# ---------------------------------------------------------------------------
# Document quality assessment
# ---------------------------------------------------------------------------

def assess_quality(ocr_result: list, raw_text: str) -> tuple[DocumentQuality, float]:
    """
    Assess document readability from PaddleOCR output.

    Args:
        ocr_result: Raw output from PaddleOCR (list of line results).
                    Each line: [[bbox], (text, confidence)]
        raw_text:   Concatenated text from all lines.

    Returns:
        (DocumentQuality, avg_confidence)
    """
    if not ocr_result or not raw_text.strip():
        return DocumentQuality.EMPTY, 0.0

    confidences: list[float] = []
    for line in ocr_result:
        try:
            # PaddleOCR format: [bbox, (text, conf)]
            conf = float(line[1][1])
            confidences.append(conf)
        except (IndexError, TypeError, ValueError):
            continue

    if not confidences:
        return DocumentQuality.UNREADABLE, 0.0

    avg_conf = sum(confidences) / len(confidences)
    word_count = len(raw_text.split())

    if avg_conf < OCR_READABILITY_THRESHOLD or word_count < 5:
        return DocumentQuality.UNREADABLE, avg_conf
    elif avg_conf < 0.60 or word_count < 20:
        return DocumentQuality.LOW, avg_conf
    else:
        return DocumentQuality.GOOD, avg_conf


# ---------------------------------------------------------------------------
# Main verifier class
# ---------------------------------------------------------------------------

class DocumentVerifier:
    """
    Agent 1 — Document Verifier.

    Usage:
        from app.agents.doc_verifier import DocumentVerifier
        verifier = DocumentVerifier()
        result = verifier.verify(
            ocr_texts  = {"file_001": ("raw ocr text", raw_paddle_output)},
            file_names = {"file_001": "prescription.jpg"},
            category   = "CONSULTATION",
            member_name = "Rajesh Kumar",
        )
    """

    def __init__(self) -> None:
        self._policy = get_policy()

    def verify(
        self,
        ocr_texts:   dict[str, tuple[str, list]],  # file_id → (raw_text, paddle_lines)
        file_names:  dict[str, str],                # file_id → original filename
        category:    str,
        member_name: str | None = None,
    ) -> DocVerificationResult:
        """
        Run all three verification checks.

        Args:
            ocr_texts:   Mapping of file_id to (raw_text, paddle_ocr_lines).
                         raw_text is the concatenated text; paddle_lines is the
                         raw PaddleOCR output used for quality assessment.
            file_names:  Mapping of file_id to the original upload filename.
            category:    Claim category string (e.g. "CONSULTATION").
            member_name: Member name from policy roster for name-match check.

        Returns:
            DocVerificationResult — check result.passed before continuing.
        """
        try:
            return self._run_verification(ocr_texts, file_names, category, member_name)
        except Exception as exc:
            logger.exception("DocumentVerifier crashed: %s", exc)
            return DocVerificationResult(
                passed       = False,
                errors       = [
                    "Document verification encountered an unexpected error. "
                    "Please try again or contact support."
                ],
                agent_status = AgentStatus.FAILED,
                agent_error  = str(exc),
            )

    def _run_verification(
        self,
        ocr_texts:   dict[str, tuple[str, list]],
        file_names:  dict[str, str],
        category:    str,
        member_name: str | None,
    ) -> DocVerificationResult:

        errors:          list[str]                    = []
        classifications: list[DocumentClassification] = []

        # ── Step 1: Classify each document + assess quality ──────────────────
        for file_id, (raw_text, paddle_lines) in ocr_texts.items():
            fname    = file_names.get(file_id, file_id)
            quality, ocr_conf = assess_quality(paddle_lines, raw_text)
            doc_type, cls_conf = classify_text(raw_text)
            patient_name       = _extract_name_from_text(raw_text)

            # Overall confidence = average of quality signal and classification signal
            combined_conf = (ocr_conf + cls_conf) / 2.0

            classifications.append(DocumentClassification(
                file_id       = file_id,
                file_name     = fname,
                classified_as = doc_type,
                quality       = quality,
                patient_name  = patient_name,
                confidence    = round(combined_conf, 3),
                notes         = self._classification_notes(doc_type, quality, ocr_conf),
            ))

        # ── Step 2: Check for unreadable documents (TC002) ───────────────────
        unreadable_errors = self._check_readability(classifications)
        errors.extend(unreadable_errors)

        # If documents are unreadable we cannot classify them reliably,
        # so stop here and ask for re-upload before checking types.
        if unreadable_errors:
            return DocVerificationResult(
                passed          = False,
                errors          = errors,
                classifications = classifications,
                agent_status    = AgentStatus.SUCCESS,
            )

        # ── Step 3: Check document types vs requirements (TC001) ─────────────
        type_errors = self._check_document_types(classifications, category)
        errors.extend(type_errors)

        # ── Step 4: Check patient name consistency (TC003) ───────────────────
        name_errors = self._check_patient_names(classifications, member_name)
        errors.extend(name_errors)

        return DocVerificationResult(
            passed          = len(errors) == 0,
            errors          = errors,
            classifications = classifications,
            agent_status    = AgentStatus.SUCCESS,
        )

    # ── Check helpers ────────────────────────────────────────────────────────

    def _check_readability(
        self, classifications: list[DocumentClassification]
    ) -> list[str]:
        """TC002: Flag documents that are too blurry/low-quality to read."""
        errors: list[str] = []
        for c in classifications:
            if c.quality == DocumentQuality.EMPTY:
                errors.append(
                    f"The file '{c.file_name}' appears to be blank or empty. "
                    f"Please re-upload a clear, legible photo or scan of your document."
                )
            elif c.quality == DocumentQuality.UNREADABLE:
                errors.append(
                    f"The file '{c.file_name}' is too blurry or low-quality to read "
                    f"(OCR confidence: {c.confidence:.0%}). "
                    f"Please re-upload a clearer photo — ensure good lighting, "
                    f"hold the camera steady, and make sure the entire document is visible."
                )
        return errors

    def _check_document_types(
        self,
        classifications: list[DocumentClassification],
        category: str,
    ) -> list[str]:
        """
        TC001: Check that the uploaded document types satisfy the required
        types for the claim category. Returns specific, actionable error messages.
        """
        errors: list[str] = []
        required: list[str] = self._policy.get_required_documents(category)

        if not required:
            logger.warning("No document requirements found for category: %s", category)
            return []

        # Build a set of what was actually uploaded (by classified type value)
        uploaded_types: set[str] = {c.classified_as.value for c in classifications}

        # Check each required type
        missing_required: list[str] = []
        for req_type in required:
            if req_type not in uploaded_types:
                missing_required.append(req_type)

        if not missing_required:
            return []

        # Build a descriptive error — name the uploaded types and missing types
        uploaded_display = self._format_type_list(list(uploaded_types))
        missing_display  = self._format_type_list(missing_required)

        # Check for the specific "wrong type uploaded" case (TC001):
        # member uploaded documents but none matched a required type
        all_unknown = all(c.classified_as == DocumentType.UNKNOWN for c in classifications)

        if all_unknown:
            errors.append(
                f"We could not identify any of your uploaded documents as valid "
                f"medical documents for a {category.title()} claim. "
                f"Please upload: {missing_display}."
            )
        else:
            # Name exactly what was found and what is still needed
            errors.append(
                f"Your {category.title()} claim requires: {self._format_type_list(required)}. "
                f"You uploaded: {uploaded_display}. "
                f"Still missing: {missing_display}. "
                f"Please upload the missing document(s) to proceed."
            )

            # Additional per-file detail for substitution errors (e.g. two prescriptions)
            for req_type in missing_required:
                # Find files classified as something other than what was needed
                wrong_files = [
                    c for c in classifications
                    if c.classified_as.value != req_type
                    and c.classified_as != DocumentType.UNKNOWN
                    and c.classified_as.value not in required
                ]
                for wf in wrong_files:
                    errors.append(
                        f"  • '{wf.file_name}' was identified as a "
                        f"{self._friendly_type(wf.classified_as.value)}, "
                        f"but a {self._friendly_type(req_type)} is required instead."
                    )

        return errors

    def _check_patient_names(
        self,
        classifications: list[DocumentClassification],
        member_name: str | None,
    ) -> list[str]:
        """
        TC003: Cross-check patient names across documents.
        Returns an error naming both the mismatched names and their source files.
        """
        errors: list[str] = []

        # Collect only documents where a name was successfully extracted
        named_docs = [c for c in classifications if c.patient_name]
        if len(named_docs) < 2:
            # Need at least 2 documents with names to detect a mismatch
            return []

        reference      = named_docs[0]
        reference_name = reference.patient_name

        for doc in named_docs[1:]:
            if not _names_match(reference_name, doc.patient_name):
                errors.append(
                    f"Patient name mismatch detected across your documents: "
                    f"'{reference.file_name}' is for '{reference_name}', "
                    f"but '{doc.file_name}' is for '{doc.patient_name}'. "
                    f"All documents in a single claim must belong to the same patient. "
                    f"Please re-upload documents that all refer to the same person."
                )
                # Report only the first mismatch found to avoid flooding the user
                break

        # Also check against the policy member name if available
        if member_name and named_docs and not errors:
            for doc in named_docs:
                if not _names_match(member_name, doc.patient_name):
                    errors.append(
                        f"The patient name on '{doc.file_name}' ('{doc.patient_name}') "
                        f"does not match the policy member name ('{member_name}'). "
                        f"If this document is for a covered dependent, please confirm "
                        f"the member ID is correct. Otherwise, re-upload the document "
                        f"for the correct patient."
                    )
                    break

        return errors

    # ── Formatting helpers ───────────────────────────────────────────────────

    @staticmethod
    def _friendly_type(doc_type_value: str) -> str:
        """Convert a DocumentType enum value to a human-readable label."""
        mapping = {
            "PRESCRIPTION":     "doctor's prescription (Rx)",
            "HOSPITAL_BILL":    "hospital/clinic bill or receipt",
            "LAB_REPORT":       "laboratory test report",
            "PHARMACY_BILL":    "pharmacy bill",
            "DENTAL_REPORT":    "dental treatment report",
            "DISCHARGE_SUMMARY": "discharge summary",
            "UNKNOWN":          "unidentified document",
        }
        return mapping.get(doc_type_value, doc_type_value.replace("_", " ").title())

    @classmethod
    def _format_type_list(cls, types: list[str]) -> str:
        """Format a list of DocumentType values into a readable string."""
        return ", ".join(cls._friendly_type(t) for t in types)

    @staticmethod
    def _classification_notes(
        doc_type: DocumentType,
        quality:  DocumentQuality,
        ocr_conf: float,
    ) -> str | None:
        parts: list[str] = []
        if quality == DocumentQuality.LOW:
            parts.append(f"low OCR confidence ({ocr_conf:.0%}) — some fields may be inaccurate")
        if doc_type == DocumentType.UNKNOWN:
            parts.append("could not identify document type from content")
        return "; ".join(parts) if parts else None
