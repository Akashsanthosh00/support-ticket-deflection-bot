from pydantic import BaseModel, Field


class TicketRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The customer's support message/question")


class SourceRef(BaseModel):
    id: str
    type: str
    title: str
    score: float


class TicketResponse(BaseModel):
    decision: str  # 'answered' | 'escalated'
    answer: str | None
    escalation_reasons: list[str]
    answer_confidence: float | None
    retrieval_top_score: float | None
    sentiment_score: float | None
    matched_keyword: str | None
    sources: list[SourceRef]


class StatsResponse(BaseModel):
    total: int
    answered: int
    escalated: int
    deflection_rate: float


class LogEntry(BaseModel):
    id: int
    created_at: str
    customer_message: str
    decision: str
    escalation_reasons: str
    answer: str | None
    answer_confidence: float | None
    retrieval_top_score: float | None
    sentiment_score: float | None
    matched_keyword: str | None
    sources: str
