# Support Ticket Deflection Bot

A customer support bot that auto-answers tickets it's confident about, and
abstains (escalates to a human) whenever it isn't. Built with FastAPI, the
Claude API, hybrid (dense + BM25) retrieval over Qdrant with cross-encoder
reranking, and a Streamlit demo UI.

## How the abstention logic works

Every incoming ticket is checked against **four independent signals**. If
**any** fires, the ticket is escalated to a human instead of auto-answered:

| # | Signal | What it catches |
|---|--------|------------------|
| 1 | **Forced keyword/topic match** | High-risk topics (refunds, legal, fraud, GDPR, account deletion, etc.) — always escalate regardless of confidence |
| 2 | **Sentiment/frustration score** (VADER) | Angry/frustrated customers who need human empathy, even for factual questions |
| 3 | **Retrieval relevance score** (hybrid search + cross-encoder rerank) | Questions with no good match in the knowledge base — bot has nothing solid to ground an answer in |
| 4 | **Claude's self-rated confidence** | Cases where retrieval looked okay but the content doesn't actually answer the question well |

This is intentionally conservative — it favors *not* giving a wrong answer
over maximizing the deflection rate. Tune the thresholds in `.env` as you
gather real data on what your team is comfortable auto-answering.

## Retrieval pipeline

1. **Dense leg** — query embedded with `sentence-transformers`, searched against a **Qdrant** collection (cosine similarity).
2. **Sparse leg** — same query scored against a **BM25** index (`rank_bm25`) for keyword overlap.
3. **Merge** — the two candidate sets are unioned.
4. **Rerank** — a **cross-encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) scores each (query, candidate) pair directly; the top results by that score go to the LLM. The cross-encoder score is also what feeds signal #3 above.

## Project structure

```
deflection-bot/
├── app/
│   ├── main.py            # FastAPI app + routes (pure JSON API)
│   ├── abstention.py       # Core decision logic (the heart of the project)
│   ├── retrieval.py         # Hybrid (Qdrant + BM25) search + cross-encoder rerank
│   ├── sentiment.py          # VADER-based frustration detection
│   ├── llm.py                  # Claude API wrapper for answer generation
│   ├── db.py                    # SQLite logging of every ticket + decision
│   ├── schemas.py                 # Pydantic request/response models
│   └── config.py                    # All tunable settings (from .env)
├── streamlit_app.py           # Streamlit demo UI, calls the FastAPI backend
├── data/
│   ├── faqs.json               # FAQ knowledge base (seed data — regenerate with scripts/collect_data.py --docs)
│   └── past_tickets.json        # Resolved tickets (seed data — regenerate with scripts/collect_data.py --issues)
├── scripts/
│   ├── collect_data.py            # Pull issues (PyGithub) + clone docs (GitPython) into data/
│   ├── build_index.py              # (Re)build the Qdrant + BM25 index after editing data/
│   ├── eval.py                      # Decision accuracy + LLM-judge false-answer rate
│   └── eval_ragas.py                 # RAGAS: faithfulness, answer relevance, context relevance
├── tests/
│   └── test_abstention.py             # Unit tests for keyword/sentiment logic
└── requirements.txt
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure your API key:**
   ```bash
   cp .env.example .env
   # then edit .env and set ANTHROPIC_API_KEY=sk-ant-...
   ```

3. **Run the backend:**
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
   The first request builds the Qdrant + BM25 index automatically (downloads
   the embedding + reranker models the first time — needs internet access).
   Qdrant runs in local on-disk mode by default, so no separate server is
   required unless you set `QDRANT_URL`. Note: local mode file-locks its
   storage folder to one process at a time — fine for a single `uvicorn`
   process, but don't run multiple `--workers` against local mode; set
   `QDRANT_URL` (e.g. Qdrant Cloud, or `docker run qdrant/qdrant`) if you
   need more than one process talking to it.

4. **Run the demo UI** (separate terminal):
   ```bash
   streamlit run streamlit_app.py
   ```
   Open the URL Streamlit prints, type a message, and see whether it's
   answered or escalated — with reasons and confidence scores shown.

## API

- `POST /ticket` — `{"message": "..."}` → returns decision, answer (if any), reasons, scores, sources
- `GET /stats` — aggregate deflection rate stats
- `GET /logs?limit=50` — recent ticket log
- `POST /reindex` — rebuild the knowledge index after editing `data/*.json`
- `GET /health` — health check

Interactive API docs (Swagger) are auto-generated at http://localhost:8000/docs

## Collecting real data

```bash
# in .env: set GITHUB_TOKEN, GITHUB_REPO (for issues), DOCS_GIT_URL (for docs)
python -m scripts.collect_data --issues --docs
python -m scripts.build_index
```

`--issues` pulls closed, commented GitHub issues from `GITHUB_REPO` into
`data/past_tickets.json` (question = issue body, resolution = top comment).
`--docs` clones `DOCS_GIT_URL` and chunks its markdown files by heading into
`data/faqs.json`. The JSON shape stays compatible with the hand-written seed
data, so you can mix collected and manually-curated entries.

## Running the evals

```bash
python -m scripts.eval          # decision accuracy, deflection rate, LLM-judge false-answer rate
python -m scripts.eval_ragas    # RAGAS: faithfulness, answer relevance, context precision/recall
```

`eval.py` covers the abstention-specific metrics that are unique to this
project. `eval_ragas.py` covers the generic RAG quality metrics via the
RAGAS library.

## Running unit tests

```bash
pytest tests/
```

These test the keyword-matching and sentiment logic directly, without
needing an API key or the embedding/reranker models — fast to run in CI.

## Customizing for your data

1. Replace `data/faqs.json` and `data/past_tickets.json` — either edit them
   directly or regenerate via `scripts/collect_data.py` (same JSON shape either way).
2. Run `python -m scripts.build_index` to rebuild the index (or hit `POST /reindex`).
3. Adjust `FORCE_ESCALATE_KEYWORDS` in `.env` to match your own high-risk topics.
4. Tune `MIN_RETRIEVAL_SIMILARITY`, `MIN_ANSWER_CONFIDENCE`, and
   `FRUSTRATION_SENTIMENT_THRESHOLD` based on eval results — start
   conservative (current defaults) and loosen carefully as you build trust
   in the auto-answer quality.

## Deployment

- **Backend**: host `app/main.py` anywhere that runs Python (a VM, Render,
  Railway, Fly.io, etc.) with `uvicorn app.main:app`. Point `QDRANT_URL` at a
  managed Qdrant instance (e.g. Qdrant Cloud) if you need it to survive
  restarts/scale beyond one process.
- **Frontend**: push this repo to **GitHub**, then deploy `streamlit_app.py`
  on **Streamlit Community Cloud** (free) pointed at your GitHub repo. Set
  `API_BASE_URL` in the app's Streamlit secrets to your hosted backend URL.

## What's NOT included (next steps for production)

- **Real ticketing system integration** (Zendesk/Intercom/etc.) — you'd add
  a webhook receiver and a client for your ticketing API's "reply" and
  "escalate to queue" actions.
- **Auth** on the API endpoints — add an API key or OAuth layer before
  exposing this beyond localhost.
- **A proper database** — SQLite is fine for testing/small volume; swap for
  Postgres for production scale.
- **Human feedback loop** — logging whether a human agent agreed with an
  auto-answer (or had to correct/reverse an escalation) so thresholds can be
  tuned from real outcomes over time.
