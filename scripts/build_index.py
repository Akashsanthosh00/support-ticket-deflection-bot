"""
Standalone script to (re)build the knowledge index (Qdrant + BM25) from
data/faqs.json and data/past_tickets.json. Run this after editing either
data file, or after scripts/collect_data.py refreshes them.

Usage:
    python -m scripts.build_index
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.retrieval import knowledge_index

if __name__ == "__main__":
    knowledge_index.build()
    print(f"Indexed {len(knowledge_index.items)} items "
          f"({sum(1 for i in knowledge_index.items if i['source_type']=='faq')} FAQs, "
          f"{sum(1 for i in knowledge_index.items if i['source_type']=='ticket')} past tickets) "
          f"into Qdrant + BM25.")
