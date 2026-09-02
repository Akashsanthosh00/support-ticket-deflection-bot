"""
FastAPI entry point for the ticket deflection bot.

This is a pure JSON API — the demo UI lives in streamlit_app.py (run
separately, talks to this API over HTTP). See README for how to run both.

Endpoints:
  POST /ticket    -> submit a customer message, get an answer or escalation
  GET  /stats     -> aggregate deflection stats
  GET  /logs      -> recent ticket log (for the Streamlit dashboard)
  POST /reindex   -> rebuild the knowledge index (after editing data files)
  GET  /health    -> health check
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db import init_db, log_ticket, get_recent_logs, get_stats
from app.retrieval import ensure_index_ready, knowledge_index
from app.abstention import process_ticket
from app.schemas import TicketRequest, TicketResponse, StatsResponse, SourceRef


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    ensure_index_ready()
    yield


app = FastAPI(title="Support Ticket Deflection Bot", lifespan=lifespan)

# Streamlit runs as a separate process/origin, so it needs CORS to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ticket", response_model=TicketResponse)
def submit_ticket(req: TicketRequest):
    result = process_ticket(req.message)

    log_ticket(
        customer_message=req.message,
        decision=result.decision,
        escalation_reasons=result.escalation_reasons,
        answer=result.answer,
        answer_confidence=result.answer_confidence,
        retrieval_top_score=result.retrieval_top_score,
        sentiment_score=result.sentiment_score,
        matched_keyword=result.matched_keyword,
        sources=[s["id"] for s in result.sources],
    )

    return TicketResponse(
        decision=result.decision,
        answer=result.answer,
        escalation_reasons=result.escalation_reasons,
        answer_confidence=result.answer_confidence,
        retrieval_top_score=result.retrieval_top_score,
        sentiment_score=result.sentiment_score,
        matched_keyword=result.matched_keyword,
        sources=[SourceRef(**s) for s in result.sources],
    )


@app.get("/stats", response_model=StatsResponse)
def stats():
    return StatsResponse(**get_stats())


@app.get("/logs")
def logs(limit: int = 50):
    return get_recent_logs(limit=limit)


@app.post("/reindex")
def reindex():
    knowledge_index.build()
    return {"status": "ok", "items_indexed": len(knowledge_index.items)}


@app.get("/health")
def health():
    return {"status": "ok"}
