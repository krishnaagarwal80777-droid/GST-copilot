# GST Tax Copilot

A citation-grounded assistant for Indian GST law. It answers a question as the law stood on a
reference date. Every statement cites the provision version in force on that date (section,
rule or rate notification), and it abstains when the evidence is thin. Retrieval, date
filtering, citation checks and guardrails are deterministic; the LLM only drafts claims as
JSON, and the answer is rendered from a fixed template.

## How to run

You need Python 3.11+, Docker, and a Gemini API key for the generator.

**1. Install dependencies**

```bash
uv sync --all-extras          # or pip-install the dependencies and extras in pyproject.toml
cp .env.example .env          # then fill in GEMINI_API_KEY
```

**2. Start pgvector**

```bash
docker compose up -d          # Postgres 16 + pgvector on localhost:5432
```

If Postgres is not reachable, the retriever falls back to an embedded LanceDB store under
`data/index/`. That store is empty until indexes are built into it, so check that
`GET /health` reports `"store": "pgvector"` once the API has loaded.

**3. Build the indexes** (first run only; `bge-m3` embeddings are cached on disk)

```bash
python -c "from retrieval.pipeline import build_indexes; from retrieval.store import open_store; from retrieval.embed import Embedder; print(build_indexes(open_store(), Embedder()))"
```

This needs the processed records in `data/processed/` (see the `make` targets) and the source
files listed in `data/raw/manifest.json`.

**4. Run the API**

```bash
uvicorn api.app:app --port 8000
```

- `GET /health` returns the status, whether the pipeline is loaded, and the store backend.
- `POST /ask` takes `{"question": "...", "reference_date": "YYYY-MM-DD", "knowledge_date": "YYYY-MM-DD"}`.
  `knowledge_date` is optional. The response holds the answer, the claims with their
  citation ids, numbered citations (path, version id, effective dates, source), the
  abstention reason, the guardrail that fired (if any), timings per stage, the model and a
  request id. Bad input returns 422.
- The first request loads the embedding and cross-encoder models, so expect it to take
  30 seconds or more. Later requests take a few seconds.
- Interactive docs are at http://localhost:8000/docs.

```bash
curl -s -X POST localhost:8000/ask -H "content-type: application/json" \
  -d '{"question": "What documents are required to claim input tax credit under rule 36?", "reference_date": "2025-06-01"}'
```

**5. Run the UI** (in a second terminal, with the API running)

```bash
streamlit run ui/app.py       # GST_API_URL overrides http://localhost:8000
```

The Ask tab has a question box, a reference-date picker and an optional knowledge date. It
shows the answer with numbered citations that expand to the provision text and its effective
dates. The Observability tab summarises the request log.

**Observability**

Every request appends one JSON line to `logs/requests.jsonl`; set `GST_REQUEST_LOG` to change
the path. A line holds a hash of the question, never its text, and no prompts or API keys.

```bash
python -m observability.report   # p50/p95 latency per stage, abstention and guardrail rates, tokens and cost
```

Costs are estimated at list price from the table in `observability/config.py`.

**Tests and evals**

```bash
pytest                           # the API tests use a stub generator: no LLM calls
python -m eval.guardrails        # guardrail refusal precision/recall (no LLM calls)
```
