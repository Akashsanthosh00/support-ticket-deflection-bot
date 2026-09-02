"""
Unit tests for the parts of the abstention pipeline that don't require an
Anthropic API key (keyword matching, sentiment scoring). Run with:

    pytest tests/

The full end-to-end pipeline (including LLM calls) is exercised by
scripts/eval.py instead, since that requires a real API key.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.abstention import check_forced_keywords
from app.sentiment import get_sentiment_score, is_frustrated


def test_forced_keyword_matches():
    assert check_forced_keywords("I want a refund please") == "refund"
    assert check_forced_keywords("This is fraud, unauthorized charge on my card") in (
        "fraud", "unauthorized charge"
    )
    assert check_forced_keywords("How do I reset my password?") is None


def test_forced_keyword_case_insensitive():
    assert check_forced_keywords("I need a REFUND now") == "refund"


def test_sentiment_negative():
    score = get_sentiment_score("This is absolutely terrible, I hate this, worst experience ever")
    assert score < 0


def test_sentiment_positive():
    score = get_sentiment_score("Thanks so much, this is great, I love it!")
    assert score > 0


def test_is_frustrated_threshold():
    assert is_frustrated("This is garbage, I'm furious and this is the worst thing ever", threshold=-0.4)
    assert not is_frustrated("How do I reset my password?", threshold=-0.4)


if __name__ == "__main__":
    test_forced_keyword_matches()
    test_forced_keyword_case_insensitive()
    test_sentiment_negative()
    test_sentiment_positive()
    test_is_frustrated_threshold()
    print("All tests passed.")
