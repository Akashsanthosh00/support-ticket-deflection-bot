import csv
import json
import random
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

random.seed(42)

RAW = Path("data/tickets_raw.json")
DOCS = Path("data/supabase_docs.jsonl")
OUT = Path("eval/label_sheet.csv")

QUOTA = {"Questions": 60, "Troubleshooting": 15, "Feature Requests": 12, "General": 8}
HOLDOUT_SIZE = 20


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def main():
    OUT.parent.mkdir(exist_ok=True)

    docs = [json.loads(line) for line in DOCS.read_text(encoding="utf-8").splitlines()]
    bm25 = BM25Okapi([tokenize(d["title"] + " " + d["text"]) for d in docs])

    raw = json.loads(RAW.read_text(encoding="utf-8"))
    by_cat = {}
    for it in raw:
        cat = (it.get("category") or {}).get("name")
        body = (it.get("body") or "").strip()
        if cat in QUOTA and len(body) >= 80:
            by_cat.setdefault(cat, []).append(it)

    picked = []
    for cat, n in QUOTA.items():
        pool = by_cat.get(cat, [])
        random.shuffle(pool)
        picked.extend(pool[:n])
    random.shuffle(picked)

    rows = []
    for i, it in enumerate(picked):
        text = (it["title"] + "\n\n" + it["body"])[:1500]
        scores = bm25.get_scores(tokenize(it["title"] + " " + it["body"][:500]))
        top = sorted(range(len(docs)), key=lambda j: scores[j], reverse=True)[:3]
        answer = (it.get("answer") or {}).get("body") or ""
        rows.append({
            "id": f"T{i + 1:03d}",
            "category": it["category"]["name"],
            "ticket_text": text,
            "source_url": it["url"],
            "accepted_answer_excerpt": answer[:600],
            "split": "holdout" if i < HOLDOUT_SIZE else "tune",
            "suggested_docs": " | ".join(docs[j]["id"] for j in top),
            "should_answer": "",
            "reference_answer": "",
            "doc_id": "",
        })

    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} tickets to {OUT}")


if __name__ == "__main__":
    main()