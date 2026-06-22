# Health Insurance Claims Processing System

Automated claims adjudication system for Plum's Group Health Insurance policy.
Processes OPD claims end-to-end: document verification → OCR extraction → policy
adjudication → fraud detection → explainable decision.

---

## Stack

| Layer | Choice | Reason |
|---|---|---|
| Web framework | FastAPI | Async, typed, auto-docs |
| OCR | PaddleOCR | Best open-source accuracy on Indian documents |
| Extraction | Regex + rule patterns | Deterministic, auditable, no LLM latency |
| Policy logic | Pure Python | Rules from JSON, no inference needed |
| Database | SQLite + SQLAlchemy | Zero-dependency, sufficient for assignment scale |
| Frontend | Single HTML file | No build step, easy to inspect |

---

## Local Setup (without Docker)

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install system dependency (macOS)
brew install poppler

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Set environment variables (or use defaults)
export DATABASE_URL="sqlite:///./data/claims.db"
export UPLOAD_DIR="./uploads"
export POLICY_PATH="./policy_terms.json"

# 5. Create runtime directories
mkdir -p data uploads

# 6. Run
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000 for the UI, http://localhost:8000/docs for the API explorer.

---

## Docker Setup

```bash
# Build and start
docker-compose up --build

# Stop
docker-compose down

# Stop and remove all data
docker-compose down -v
```

---

## Running the Eval Suite

```bash
# With the server running locally:
python tests/run_eval.py

# Output: tests/eval_report.json
```

---

## Architecture

```
POST /claims/submit
        │
        ▼
┌───────────────────────────────────┐
│         Orchestrator              │  main.py
│  Runs agents 1-4 in sequence.     │
│  Catches exceptions per-agent.    │
│  Assembles ClaimResult.           │
│  Writes to SQLite.                │
└─────┬──────┬──────┬───────┬───────┘
      │      │      │       │
      ▼      ▼      ▼       ▼
  Agent1  Agent2  Agent3  Agent4
  Doc     OCR     Policy  Fraud
  Verify  Extract Engine  Detect
```

### Agent 1 — Document Verifier (`agents/doc_verifier.py`)

Runs before any OCR. Classifies each uploaded file using keyword/layout
heuristics, then checks:

1. Do the classified types satisfy the required types for the claim category?
2. Is every document readable (not a blank or corrupted image)?
3. Do patient names extracted from different documents match?

If any check fails, the pipeline stops immediately with a specific, actionable
error message. No generic errors.

### Agent 2 — OCR Extractor (`agents/ocr_extractor.py`)

Converts each file to images (PDF → pages via pdf2image, images directly).
Runs PaddleOCR to get raw text. Applies regex and pattern matching to extract:
patient name, doctor name, registration number, diagnosis, date, line items,
total amount. Assigns a per-field confidence and an overall document confidence.

### Agent 3 — Policy Engine (`agents/policy_engine.py`)

Pure Python. Reads all rules from `policy_terms.json` via `config.py`. Runs
checks in a defined order and accumulates a trace. Each check produces a
`CheckResult(name, passed, detail)`. The final decision, approved amount, and
complete trace all come from this agent.

Checks (in order):
1. Member exists
2. 30-day initial waiting period
3. Condition-specific waiting period
4. Global exclusions
5. Pre-authorization requirement
6. Category sub-limit
7. Per-claim hard limit
8. Annual OPD limit (YTD)
9. Line-item level inclusion/exclusion
10. Network discount → co-pay calculation

### Agent 4 — Fraud Detector (`agents/fraud_detector.py`)

Queries SQLite for claim history. Checks same-day claim count, monthly count,
and high-value threshold. Computes a fraud score. If the score exceeds the
threshold in `policy_terms.json`, upgrades the decision to `MANUAL_REVIEW`.

### Graceful Degradation

Every agent call in the orchestrator is wrapped in a try/except. If an agent
raises an unhandled exception, the orchestrator logs it, marks that agent as
`FAILED` in the trace, reduces the confidence score, and continues with the
remaining agents. The final result always returns a decision — never a 500
crash. (This satisfies TC011.)

---

## Design Decisions and Trade-offs

**No LLM for extraction.** PaddleOCR + regex is deterministic and auditable.
An LLM would add latency, cost, hallucination risk, and non-determinism to a
compliance-sensitive pipeline. The regex patterns cover the document formats
described in `sample_documents_guide.md`.

**No task queue.** For the assignment scale (single user, test runner), a
synchronous pipeline in a single FastAPI request is fine. At 10x load, the
natural upgrade is to move to a background job queue (Celery + Redis) and
return a `claim_id` immediately, polling for results.

**SQLite.** Sufficient for this assignment. At production scale, the
SQLAlchemy models require no changes — swap the `DATABASE_URL` to Postgres.

**Single HTML file.** No build step, no Node, easy to audit. At product scale,
this becomes a React app that talks to the same API.

**Agents are classes, not microservices.** In-process communication is
simpler, faster, and easier to test. The interface contract (input/output
Pydantic models) is the boundary — each agent can be extracted to a
microservice without changing its contract.

---

## Limitations

- Handwritten prescriptions will have lower extraction accuracy. The system
  reflects this in per-field confidence scores and surface it in the trace.
- Regex patterns cover common Indian medical document formats but will miss
  unusual layouts. Fields that fail extraction are marked `LOW_CONFIDENCE`
  rather than causing a hard failure.
- SQLite has no concurrent write support. Under concurrent load, use Postgres.
- PaddleOCR model download (~200 MB) happens at Docker build time. First build
  is slow; subsequent builds use the layer cache.

---

## File Map

```
claims-processing/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
├── policy_terms.json          ← provided, read by config.py
├── test_cases.json            ← provided, used by tests/run_eval.py
│
├── app/
│   ├── main.py                ← FastAPI app, orchestrator, all routes
│   ├── config.py              ← policy loader, typed accessors
│   ├── models.py              ← Pydantic + SQLAlchemy models
│   ├── database.py            ← SQLite engine, session factory
│   └── agents/
│       ├── doc_verifier.py    ← Agent 1
│       ├── ocr_extractor.py   ← Agent 2
│       ├── policy_engine.py   ← Agent 3
│       └── fraud_detector.py  ← Agent 4
│
├── static/
│   └── index.html             ← submission UI + decision review
│
├── tests/
│   ├── run_eval.py            ← runs all 12 test cases, writes report
│   └── test_agents.py         ← unit tests per agent
│
└── uploads/                   ← runtime, gitignored
```
