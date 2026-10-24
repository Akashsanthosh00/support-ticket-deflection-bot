import json
import os
import time
from pathlib import Path

import requests

TOKEN = os.environ["GITHUB_TOKEN"]
OUT_FILE = Path("data/tickets_raw.json")
TARGET = 400

QUERY = """
query($cursor: String) {
  repository(owner: "supabase", name: "supabase") {
    discussions(first: 50, after: $cursor,
                orderBy: {field: CREATED_AT, direction: DESC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number
        title
        body
        url
        createdAt
        category { name }
        answer { body }
        comments { totalCount }
      }
    }
  }
}
"""


def fetch_page(cursor):
    resp = requests.post(
        "https://api.github.com/graphql",
        json={"query": QUERY, "variables": {"cursor": cursor}},
        headers={"Authorization": f"bearer {TOKEN}"},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        raise RuntimeError(data["errors"])
    return data["data"]["repository"]["discussions"]


def main():
    OUT_FILE.parent.mkdir(exist_ok=True)
    items, cursor = [], None
    while len(items) < TARGET:
        page = fetch_page(cursor)
        items.extend(page["nodes"])
        print(f"fetched {len(items)}")
        if not page["pageInfo"]["hasNextPage"]:
            break
        cursor = page["pageInfo"]["endCursor"]
        time.sleep(1)
    OUT_FILE.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Saved {len(items)} discussions to {OUT_FILE}")
    cats = {}
    for it in items:
        name = (it.get("category") or {}).get("name", "none")
        cats[name] = cats.get(name, 0) + 1
    print("By category:", cats)
    print("With accepted answer:", sum(1 for it in items if it.get("answer")))


if __name__ == "__main__":
    main()