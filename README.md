# Groww Fund Facts

A facts-only assistant for a small, curated set of HDFC Mutual Fund schemes, branded for Groww. It answers from **15 curated Groww.in pages** (HDFC AMC family only — see [`SOURCES.md`](SOURCES.md) / [`sources.csv`](sources.csv)), cites exactly one allowlisted Groww link, shows the source date, and refuses investment advice, performance comparisons, personal data, and out-of-corpus questions.

> Facts-only. Public Groww sources only. No advice, performance claims, or PII.

LLM provider and model stay in `.env` / Streamlit secrets (`GROQ_API_KEY`, `LLM_MODEL_ID`). Do not commit secrets.

## Product constraints (enforced)

- **Public sources only.** Corpus is 15 curated Groww.in HDFC pages (15–25 URL band). No blogs, no app back-end screenshots. Performance refusals link to the official HDFC factsheet catalogue (link only, not ingested).
- **No PII.** PAN, Aadhaar, phone, email, OTP, and folio/account numbers are blocked before any model call; they are not logged or cached.
- **No performance claims.** Returns/CAGR/ranking questions are refused and pointed to the official factsheet.
- **Clarity.** Answers are capped at three sentences and always show `Last updated from sources: …`.

## What it does

- Answers factual questions about HDFC Large Cap Fund, HDFC Flexi Cap Fund, HDFC ELSS Tax Saver, and HDFC Mid Cap Fund.
- Covers facts such as exit load, lock-in, minimum SIP, expense ratio, riskometer, and benchmark from Groww scheme pages.
- Cites one allowlisted Groww URL with last-updated date and a verbatim proof passage.
- Routes advice, performance, PII, injection, ambiguous, and out-of-corpus questions to fixed safe responses.
- Serves a one-page Streamlit UI and a JSON API.

## What it does not do

- It does not give buy, sell, hold, or recommendation advice.
- It does not compute, compare, or predict returns or performance.
- It does not answer for schemes outside the curated corpus.
- It does not accept or store PAN, Aadhaar, phone, email, OTP, folio, or account numbers.

## Architecture

```
streamlit_app.py    Streamlit frontend (Groww white + blue)
ui/                 One-page chat UI (FastAPI)
src/api/main.py     FastAPI app and guard order
src/services/       Shared question processing and knowledge-base init
src/safety/         PII detection, intent routing, fixed responses
src/rag/            Retrieval, prompt generation, validation, cache, pipeline
src/ingest/         Source loading, normalization, chunking, embeddings, Chroma index
eval/               Gold questions and retrieval/QA/safety runner
tests/              Unit and API tests
sources.csv         Curated source inventory (Groww + HDFC AMC + SEBI + AMFI)
data/artifacts/     Generated corpus, Chroma index, and cache (gitignored)
data/eval/          Generated evaluation results
```

The request path is: PII check -> intent routing -> retrieval -> generation -> deterministic validation -> cache. Only validated factual answers are cached. Both the FastAPI and Streamlit frontends use `src/services/chat.py`, so the guard order and response envelopes are identical.

## Setup

Python 3.11+ is supported. The dependency set in `requirements.txt` is pinned and tested on **Python 3.13.0** (Windows). For Streamlit Community Cloud, select **Python 3.13** in the app's **Advanced settings**; Community Cloud does not honor `runtime.txt` for this. Python 3.12 is expected to work with these pins, but the verified test run is on 3.13.0.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

Use `requirements.txt` for deployment and `requirements-dev.txt` when you also need `pytest`.

Create a `.env` file in the project root (keep your existing key and model):

```dotenv
LLM_MODEL_ID=openai/gpt-oss-120b
GROQ_API_KEY=your_groq_key_here
```

The provider defaults to `groq` in `config.yaml`. To use another supported provider, change `llm.provider` and set the matching key (`GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `MISTRAL_API_KEY`). `.env` is gitignored; never commit keys.

## Build the corpus

The generated corpus and index live under `data/artifacts/` and are not committed. Build them before the first run (required after adding Groww sources):

```powershell
python -m src.ingest --stage all
```

Or run stages separately:

```powershell
python -m src.ingest --stage load
python -m src.ingest --stage normalize
python -m src.ingest --stage chunk
python -m src.ingest --stage embed
python -m src.ingest --stage index
```

Ingestion reads `sources.csv`, respects publisher allowlists, stores manifest and checksums, and rebuilds the Chroma collection idempotently.

## Run the app

```powershell
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/` for the UI.

## Streamlit frontend

The Streamlit app reuses the same safety, retrieval, generation, validation, and citation logic as the API.

```powershell
streamlit run streamlit_app.py
```

The app shell renders immediately. Configuration is loaded quickly, and the embedding model and knowledge base are initialized only when needed. If the corpus or index is missing, the page shows an **Initialize knowledge base** panel with stage-by-stage progress (model, download, normalize, chunk, embed, index, validate). The model and initialized resources are cached for the server process, and later reruns do not rebuild a valid Chroma index.

Answer cards show a type badge, source link (Groww preferred for scheme facts), last-updated date, collapsible proof, and an expandable **How this answer was found** panel with pipeline steps and ranked passages.

### Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (`main`).
2. In [share.streamlit.io](https://share.streamlit.io), create an app from the repo.
3. Set **Main file path** to `streamlit_app.py`.
4. In **Advanced settings**, select **Python 3.13**.
5. Add secrets in the app dashboard (same values as local `.env`):

```toml
GROQ_API_KEY = "your_groq_key_here"
LLM_MODEL_ID = "openai/gpt-oss-120b"
# CACHE_SALT = "optional-random-salt"
# HF_TOKEN = "optional-hugging-face-token"
```

6. Deploy, then open the app and click **Initialize knowledge base** on first run (rebuild after Groww sources land).

Do not commit `.streamlit/secrets.toml`; use `.streamlit/secrets.toml.example` as the template. Locally, `.env` continues to work.

Streamlit Community Cloud notes:

- Select **Python 3.13** in **Advanced settings**. `runtime.txt` is not used by Community Cloud for Python version selection. The pinned dependency set is tested on Python 3.13.0.
- After changing the Python version or `requirements.txt`, use **Manage app → Reboot** (or clear the app cache) so Community Cloud reinstalls the environment.
- `packages.txt` installs `libxml2-dev` and `libxslt1-dev` for HTML ingestion.
- `data/artifacts/` is generated and is not committed, so the first cold start downloads curated documents and the embedding model. This can take several minutes and needs outbound network access. The UI reports each initialization stage.
- The filesystem is ephemeral; a restarted or redeployed app rebuilds the corpus and index. Use a persistent external vector store if that cost matters.
- `requirements.txt` requests the CPU-only PyTorch wheel (`torch==2.15.0+cpu`) from the PyTorch CPU index. This avoids downloading the large NVIDIA CUDA packages on Community Cloud's Linux runtime. The corpus uses `device: cpu` in `config.yaml`.
- The runtime image must have enough memory for `sentence-transformers`, PyTorch, and ChromaDB.
- Streamlit Community Cloud runs the Streamlit app only. The FastAPI JSON API and the static `ui/` page remain available when deployed separately.

API endpoints:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/chat` | One response envelope for a question |
| `GET` | `/api/sources` | Curated source inventory |
| `GET` | `/api/pipeline` | Pipeline counts and sample chunks |
| `GET` | `/api/pipeline/{stage}` | Stage-level pipeline details |
| `GET` | `/health` | Health check |

Example:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/chat `
  -ContentType "application/json" `
  -Body '{"question":"What is the exit load of HDFC Flexi Cap Fund?"}'
```

## Response types

| Type | Meaning |
|---|---|
| `answer` | Validated factual answer with one link, date, and proof |
| `refusal_advice` | Investment advice was requested |
| `refusal_performance` | Returns, comparison, or ranking was requested |
| `pii` | Personal data was detected; input is not echoed or stored |
| `not_in_sources` | The fact is outside the curated corpus or failed validation |
| `clarify` | A scheme-sensitive question did not name a scheme |
| `error` | A safe fallback when a verified answer cannot be produced |

## Evaluation

The gold set is `eval/gold_questions.csv` with 27 rows: 8 factual, 3 multi-scheme, 4 advice, 3 performance, 3 PII, 2 out-of-corpus, 2 ambiguous, and 2 injection cases.

```powershell
python -m eval.run --suite all
python -m eval.run --suite retrieval
python -m eval.run --suite qa
python -m eval.run --suite safety
```

Results are written to:

- `data/eval/retrieval_eval.json`
- `data/eval/qa_eval.json`
- `data/eval/safety_eval.json`

Pass criteria:

- Safety routing 100%.
- Citation precision 100%: one real, allowlisted link per non-error response.
- Factual accuracy >= 90%.
- Retrieval recall@5 >= 0.85.
- Zero answers over three sentences.
- Zero ungrounded numbers.
- Zero PII leaks in responses, traces, cache keys, or artifacts.

## Tests

```powershell
python -m pytest tests -q
```

The suite covers loading, normalization, chunking, embeddings, the Chroma store, retrieval, PII detection, intent routing, generation, validation, the pipeline, the API, and the Phase 8 guardrail edge cases.

## Configuration

Key settings in `config.yaml`:

- `retrieval.top_k` and `retrieval.score_threshold` control candidate recall.
- `retrieval.max_context_chunks` limits context passed to the model.
- `llm.provider`, `llm.model`, temperature, and timeout control generation (defaults: Groq + `LLM_MODEL_ID` from `.env`).
- `safety.detect_pii`, `safety.max_repair_retries`, and `safety.cache_answers` control guardrails.
- `app.max_answer_sentences` is enforced by validation.
- `ingestion.allowlist_publishers` is Groww-only; `sources.csv` / `SOURCES.md` list exactly 15 Groww HDFC URLs.

`llm.model` is read from `LLM_MODEL_ID` in `.env`.

## Safety notes

- PII detection runs in plain code before any model call.
- PII questions return a fixed warning and are not logged or cached.
- Retrieved document text is treated as data, never instructions.
- The model gets only retrieved context and must cite one URL from that context.
- Deterministic validation checks sentence count, URLs, advice/performance phrases, number grounding, verbatim proof, and PII echo.
- Invalid output gets one repair attempt, then a safe `not_in_sources` response.
- The cache key is hashed and contains only the question and optional scheme.

## Disclaimer

This is a prototype for factual information only. It is not investment, tax, or legal advice. Always verify details in the latest scheme documents before acting.
