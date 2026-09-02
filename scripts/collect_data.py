"""
Data collection: pulls real Q&A pairs and docs from GitHub to build
data/past_tickets.json and data/faqs.json.

Two sources:
  1. GitHub Issues (via PyGithub) — closed issues with comments become
     "past tickets" (question = issue body, resolution = top comment).
  2. A docs repo (cloned via GitPython) — markdown files are chunked by
     heading into "FAQ" entries.

Configure via .env:
    GITHUB_TOKEN=ghp_...
    GITHUB_REPO=owner/repo          # issue source
    GITHUB_ISSUE_LABEL=question     # optional filter
    MAX_ISSUES_TO_COLLECT=200
    DOCS_GIT_URL=https://github.com/owner/docs-repo.git
    DOCS_GLOB=**/*.md

Usage:
    python -m scripts.collect_data --issues     # rebuild data/past_tickets.json
    python -m scripts.collect_data --docs        # rebuild data/faqs.json
    python -m scripts.collect_data --issues --docs
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings


def collect_issues() -> list[dict]:
    """Pull closed, commented issues from GITHUB_REPO into the past_tickets.json shape."""
    from github import Github

    if not settings.GITHUB_TOKEN or not settings.GITHUB_REPO:
        raise SystemExit(
            "Set GITHUB_TOKEN and GITHUB_REPO in .env before collecting issues."
        )

    gh = Github(settings.GITHUB_TOKEN)
    repo = gh.get_repo(settings.GITHUB_REPO)

    kwargs = {"state": "closed"}
    issues = repo.get_issues(**kwargs)

    tickets = []
    for issue in issues:
        if issue.pull_request is not None:
            continue  # skip PRs, get_issues() returns both
        if settings.GITHUB_ISSUE_LABEL and settings.GITHUB_ISSUE_LABEL not in [
            l.name for l in issue.labels
        ]:
            continue
        comments = list(issue.get_comments())
        if not comments:
            continue  # need at least one comment to treat as a "resolution"

        tickets.append(
            {
                "id": f"gh-issue-{issue.number}",
                "subject": issue.title,
                "question": (issue.body or "").strip()[:2000],
                "resolution": comments[0].body.strip()[:2000],
            }
        )
        if len(tickets) >= settings.MAX_ISSUES_TO_COLLECT:
            break

    return tickets


def _chunk_markdown(text: str, source_file: str) -> list[dict]:
    """Split a markdown file into chunks on '## ' headings."""
    chunks = re.split(r"\n(?=## )", text)
    out = []
    for i, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if not chunk or len(chunk) < 40:
            continue
        first_line = chunk.splitlines()[0].lstrip("#").strip()
        out.append(
            {
                "id": f"doc-{os.path.basename(source_file)}-{i}",
                "title": first_line or os.path.basename(source_file),
                "content": chunk,
            }
        )
    return out


def collect_docs() -> list[dict]:
    """Clone DOCS_GIT_URL and chunk its markdown files into the faqs.json shape."""
    import glob
    import git

    if not settings.DOCS_GIT_URL:
        raise SystemExit("Set DOCS_GIT_URL in .env before collecting docs.")

    if os.path.isdir(os.path.join(settings.DOCS_CLONE_DIR, ".git")):
        repo = git.Repo(settings.DOCS_CLONE_DIR)
        repo.remotes.origin.pull()
    else:
        os.makedirs(settings.DOCS_CLONE_DIR, exist_ok=True)
        git.Repo.clone_from(settings.DOCS_GIT_URL, settings.DOCS_CLONE_DIR, depth=1)

    faqs = []
    pattern = os.path.join(settings.DOCS_CLONE_DIR, settings.DOCS_GLOB)
    for path in glob.glob(pattern, recursive=True):
        with open(path, encoding="utf-8", errors="ignore") as f:
            text = f.read()
        faqs.extend(_chunk_markdown(text, path))

    return faqs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--issues", action="store_true", help="Collect GitHub issues -> past_tickets.json")
    parser.add_argument("--docs", action="store_true", help="Collect docs repo -> faqs.json")
    args = parser.parse_args()

    if not args.issues and not args.docs:
        parser.error("Pass --issues and/or --docs")

    if args.issues:
        tickets = collect_issues()
        with open(settings.TICKETS_PATH, "w") as f:
            json.dump(tickets, f, indent=2)
        print(f"Wrote {len(tickets)} tickets to {settings.TICKETS_PATH}")

    if args.docs:
        faqs = collect_docs()
        with open(settings.FAQ_PATH, "w") as f:
            json.dump(faqs, f, indent=2)
        print(f"Wrote {len(faqs)} FAQ chunks to {settings.FAQ_PATH}")

    if args.issues or args.docs:
        print("Run `python -m scripts.build_index` (or POST /reindex) to re-embed the new data.")


if __name__ == "__main__":
    main()
