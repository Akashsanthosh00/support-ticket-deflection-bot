"""
Central configuration for the deflection bot.
All tunable abstention thresholds and infra settings live here, loaded from
environment variables so they can be adjusted without touching code (see
.env.example).
"""
import os
from dotenv import load_dotenv

load_dotenv()


def _float_env(name: str, default: float) -> float:
    val = os.getenv(name)
    return float(val) if val not in (None, "") else default


def _list_env(name: str, default: list[str]) -> list[str]:
    val = os.getenv(name)
    if not val:
        return default
    return [item.strip().lower() for item in val.split(",") if item.strip()]


class Settings:
    # --- API ---
# --- LLM provider ---
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini").lower()

    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # --- Abstention thresholds ---
    # NOTE: MIN_RETRIEVAL_SIMILARITY now compares against the sigmoid-normalized
    # cross-encoder rerank score (0-1), not raw cosine similarity, since the
    # reranker score is what actually reaches the abstention decision now.
    MIN_RETRIEVAL_SIMILARITY: float = _float_env("MIN_RETRIEVAL_SIMILARITY", 0.45)
    MIN_ANSWER_CONFIDENCE: float = _float_env("MIN_ANSWER_CONFIDENCE", 0.6)
    FRUSTRATION_SENTIMENT_THRESHOLD: float = _float_env(
        "FRUSTRATION_SENTIMENT_THRESHOLD", -0.4
    )
    FORCE_ESCALATE_KEYWORDS: list[str] = _list_env(
        "FORCE_ESCALATE_KEYWORDS",
        [
            "refund", "chargeback", "lawsuit", "legal", "cancel my account",
            "fraud", "unauthorized charge", "lawyer", "sue", "gdpr",
            "delete my data",
        ],
    )
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "4"))

    # --- Retrieval / embedding ---
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # --- Hybrid retrieval (dense + BM25) ---
    HYBRID_CANDIDATE_K: int = int(os.getenv("HYBRID_CANDIDATE_K", "20"))

    # --- Reranking ---
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # --- Paths ---
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    DB_PATH: str = os.path.join(BASE_DIR, "deflection_bot.db")
    FAQ_PATH: str = os.path.join(DATA_DIR, "faqs.json")
    TICKETS_PATH: str = os.path.join(DATA_DIR, "past_tickets.json")
    INDEX_DIR: str = os.path.join(BASE_DIR, "index_store")
    BM25_INDEX_PATH: str = os.path.join(INDEX_DIR, "bm25.pkl")

    # --- Vector DB (Qdrant) ---
    # Local on-disk mode needs no running server (good for the demo/portfolio
    # deployment). Set QDRANT_URL to point at a real Qdrant server/cluster
    # instead (e.g. Qdrant Cloud) for production.
    # NOTE: local mode only supports ONE process holding the storage folder
    # at a time (it file-locks it). Fine for `uvicorn app.main:app` as a
    # single process, but `--reload`'s reloader subprocess or multiple
    # `--workers` will collide on the lock. Use `--workers 1` without
    # `--reload` for the local-mode demo, or set QDRANT_URL for anything
    # beyond a single process.
    QDRANT_URL: str = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    QDRANT_PATH: str = os.path.join(BASE_DIR, "qdrant_store")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "deflection_bot_kb")

    # --- Data collection (GitHub issues + docs repo) ---
    GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")
    GITHUB_REPO: str = os.getenv("GITHUB_REPO", "")  # e.g. "owner/repo"
    GITHUB_ISSUE_LABEL: str = os.getenv("GITHUB_ISSUE_LABEL", "")  # optional filter
    MAX_ISSUES_TO_COLLECT: int = int(os.getenv("MAX_ISSUES_TO_COLLECT", "200"))
    DOCS_GIT_URL: str = os.getenv("DOCS_GIT_URL", "")
    DOCS_CLONE_DIR: str = os.path.join(BASE_DIR, "docs_source")
    DOCS_GLOB: str = os.getenv("DOCS_GLOB", "**/*.md")

    # --- Frontend (Streamlit talks to this) ---
    API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8000")


settings = Settings()
