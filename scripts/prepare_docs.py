import json
import re
from pathlib import Path

CONTENT_ROOT = Path(r"C:\Users\HP\Desktop\supabase-source\apps\docs\content")
GUIDES_ROOT = CONTENT_ROOT / "guides"
TROUBLESHOOTING_ROOT = CONTENT_ROOT / "troubleshooting"
TOPICS = ["auth", "database", "storage", "functions"]
OUT_FILE = Path("data/supabase_docs.jsonl")

FRONTMATTER = re.compile(r"^---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)
IMPORT_LINE = re.compile(r"^\s*import .*$", re.MULTILINE)
JSX_TAG = re.compile(r"</?[A-Z][A-Za-z0-9_.]*[^>]*>|\{\s*['\"]\s['\"]\s*\}")
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


def clean(text):
    title = None
    m = FRONTMATTER.match(text)
    if m:
        t = re.search(r"^title:\s*['\"]?(.*?)['\"]?\s*$", m.group(1), re.MULTILINE)
        title = t.group(1) if t else None
        text = text[m.end():]
    text = IMPORT_LINE.sub("", text)
    text = HTML_COMMENT.sub("", text)
    text = JSX_TAG.sub(" ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return title, text


def write_records(f, root, topic, id_prefix):
    count = 0
    for path in root.rglob("*.mdx"):
        raw = path.read_text(encoding="utf-8", errors="ignore")
        title, body = clean(raw)
        if len(body) < 200:
            continue
        rel = path.relative_to(root).with_suffix("")
        record = {
            "id": f"{id_prefix}/{str(rel).replace(chr(92), '/')}",
            "title": title or path.stem,
            "topic": topic,
            "text": body,
            "url": f"https://supabase.com/docs/guides/{id_prefix}/{str(rel).replace(chr(92), '/')}",
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        count += 1
    return count


def main():
    OUT_FILE.parent.mkdir(exist_ok=True)
    total = 0
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for topic in TOPICS:
            n = write_records(f, GUIDES_ROOT / topic, topic, topic)
            print(f"{topic}: {n} docs")
            total += n
        n = write_records(f, TROUBLESHOOTING_ROOT, "troubleshooting", "troubleshooting")
        print(f"troubleshooting: {n} docs")
        total += n
    print(f"Wrote {total} docs to {OUT_FILE}")


if __name__ == "__main__":
    main()