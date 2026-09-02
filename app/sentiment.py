"""
Lightweight sentiment/frustration detection using VADER (rule-based, fast,
no model download needed). Returns a compound score from -1 (very negative)
to +1 (very positive) — this feeds into the abstention logic as one signal
for detecting frustrated customers who should be routed to a human.
"""
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def get_sentiment_score(text: str) -> float:
    """Returns VADER compound score in [-1, 1]. Lower = more negative/frustrated."""
    scores = _analyzer.polarity_scores(text)
    return scores["compound"]


def is_frustrated(text: str, threshold: float) -> bool:
    return get_sentiment_score(text) <= threshold
