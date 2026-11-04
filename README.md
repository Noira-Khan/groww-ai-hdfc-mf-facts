# HDFC MF Facts

A facts-only assistant for a small, curated set of HDFC Mutual Fund schemes. It answers from official HDFC AMC, SEBI, and AMFI sources, cites exactly one allowlisted link, shows the source date, and refuses investment advice, performance comparisons, personal data, and out-of-corpus questions.

> Facts-only. No investment advice.

## What it does

- Answers factual questions about HDFC Large Cap Fund, HDFC Flexi Cap Fund, HDFC ELSS Tax Saver, and HDFC Mid Cap Fund.
- Covers facts such as exit load, lock-in, minimum SIP, expense ratio, riskometer, benchmark, and account-statement requests.
- Returns answers with a single official source URL, the source `as_on_date`, and a verbatim proof passage.
- Routes advice, performance, PII, injection, ambiguous, and out-of-corpus questions to fixed safe responses.
- Keeps answers to at most three sentences and validates every number against retrieved context.
- Serves a one-page UI and a JSON API.

## What it does not do

- It does not give buy, sell, hold, or recommendation advice.
- It does not compute, compare, or predict returns or performance.
- It does not answer for schemes outside the curated corpus.
- It does not accept or store PAN, Aadhaar, phone, email, OTP, folio, or account numbers.

## Architecture

```
streamlit_app.py    Streamlit frontend
ui/                 One-page chat UI (FastAPI)
src/api/main.py     FastAPI app and guard order
src/services/       Shared question processing and knowledge-base init
src/safety/         PII detection, intent routing, fixed responses
src/rag/            Retrieval, prompt generation, validation, cache, pipeline
src/ingest/         Source loading, normalization, chunking, embeddings, Chroma index
eval/               Gold questions and retrieval/QA/safety runner
tests/              Unit and API tests
sources.csv         Curated official source inventory
data/artifacts/     Generated corpus, Chroma index, and cache (gitignored)
data/eval/          Generated evaluation results
```

The request path is: PII check -> intent routing -> retrieval -> generation -> deterministic validation -> cache. Only validated factual answers are cached. Both the FastAPI and Streamlit frontends use `src/services/chat.py`, so the guard order and response envelopes are identical.

## Setup

Python 3.11+ is recommended. The project has been run with Python 3.13 on Windows.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create a `.env` file in the project root:

```dotenv
LLM_MODEL_ID=openai/gpt-oss-120b
GROQ_API_KEY=your_groq_key_here
```

The provider defaults to `groq` in `config.yaml`. To use another supported provider, change `llm.provider` and set the matching key (`GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `MISTRAL_API_KEY`). `.env` is gitignored; never commit keys.

## Build the corpus

The generated corpus and index live under `data/artifacts/` and are not committed. Build them before the first run:

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

The app opens an existing, valid Chroma collection. If the corpus or index is missing, it runs the ingestion pipeline once and caches the initialized resources for the server process. Later reruns do not rebuild the index.

For Streamlit Community Cloud, add these secrets in the app dashboard:

```toml
GROQ_API_KEY = "your_groq_key_here"
LLM_MODEL_ID = "openai/gpt-oss-120b"
# CACHE_SALT = "optional-random-salt"
# HF_TOKEN = "optional-hugging-face-token"
```

Do not commit `.streamlit/secrets.toml`; use `.streamlit/secrets.toml.example` as the template. Locally, `.env` continues to work.

Streamlit Community Cloud notes:

- `data/artifacts/` is generated and is not committed, so the first cold start downloads official documents and the embedding model. This can take several minutes and needs outbound network access.
- The filesystem is ephemeral; a restarted or redeployed app rebuilds the corpus and index. Use a persistent external vector store if that cost matters.
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

The latest full run passed all three suites: retrieval recall@5 `1.0`, QA factual accuracy `1.0`, citation precision `1.0`, safety routing `1.0`, and zero PII leaks.

## Tests

```powershell
python -m pytest tests -q
```

The suite covers loading, normalization, chunking, embeddings, the Chroma store, retrieval, PII detection, intent routing, generation, validation, the pipeline, the API, and the Phase 8 guardrail edge cases.

## Configuration

Key settings in `config.yaml`:

- `retrieval.top_k` and `retrieval.score_threshold` control candidate recall.
- `retrieval.max_context_chunks` limits context passed to the model.
- `llm.provider`, `llm.model`, temperature, and timeout control generation.
- `safety.detect_pii`, `safety.max_repair_retries`, and `safety.cache_answers` control guardrails.
- `app.max_answer_sentences` is enforced by validation.

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

This is a prototype for factual information only. It is not investment, tax, or legal advice. Always verify details in the latest official scheme documents before acting.
