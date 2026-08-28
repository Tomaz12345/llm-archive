"""Phase 0 — build a retrieval evaluation set from real sessions, with no manual labelling.

The trick: every source already stores a short natural-language title for each session,
written by a model that read the conversation.

    claude_code   `ai-title` records          -> aiTitle
    codex         session_index.jsonl         -> thread_name
    opencode      storage/session/*.json      -> title

A title is almost exactly the query you would type months later ("that chat about the
offside detection"), and the session it belongs to is unambiguous ground truth. So:

    query  = session title
    target = any conversational chunk from that session
    metric = does the retriever surface the right session in the top k?

Corpus follows the §1.1 rule: user and assistant text only. Tool results are excluded,
and `thinking` blocks are excluded because Claude Code stores them empty (signature only).

Writes data/fixtures/evalset.json
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

HOME = Path.home()
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "fixtures" / "evalset.json"

MIN_CHUNK_CHARS = 40      # below this a message carries no retrievable signal
MAX_CHUNK_CHARS = 2000    # split longer messages
MIN_DOCS_PER_SESSION = 3  # a session needs some substance to be a fair target

# Slovene function words that essentially never appear in English text.
SLOVENE_HINTS = {
    "je", "in", "za", "ki", "se", "na", "da", "so", "ne", "pa", "bi", "kaj",
    "kako", "lahko", "sem", "bo", "tudi", "samo", "moram", "moraš", "želim",
    "prosim", "naredi", "nekaj", "vse", "več", "ali", "kjer", "sva", "nisem",
}


def detect_lang(text: str) -> str:
    words = set(re.findall(r"[a-zžčšćđA-ZŽČŠĆĐ]+", text.lower()))
    hits = len(words & SLOVENE_HINTS)
    if hits >= 3 or (hits >= 2 and len(words) < 25):
        return "sl"
    if re.search(r"[žčš]", text.lower()) and hits >= 1:
        return "sl"
    return "en"


def split_text(text: str) -> list[str]:
    text = text.strip()
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]
    out, buf = [], ""
    for para in text.split("\n\n"):
        if len(buf) + len(para) + 2 > MAX_CHUNK_CHARS and buf:
            out.append(buf.strip())
            buf = para
        else:
            buf = f"{buf}\n\n{para}" if buf else para
    if buf.strip():
        out.append(buf.strip())
    return [c for c in out if c]


def clean(text: str) -> str:
    """Drop harness noise that is not really conversation.

    The IDE tags matter more than they look: every message in an editor session carries
    the open file's absolute path, so dozens of unrelated sessions share the same path
    tokens. Left in, they inflate keyword scores and blur embeddings alike.
    """
    text = re.sub(r"<ide_opened_file>.*?</ide_opened_file>", " ", text, flags=re.S)
    text = re.sub(r"<ide_selection>.*?</ide_selection>", " ", text, flags=re.S)
    text = re.sub(r"<ide_diagnostics>.*?</ide_diagnostics>", " ", text, flags=re.S)
    text = re.sub(r"<command-name>.*?</command-name>", " ", text, flags=re.S)
    text = re.sub(r"<command-message>.*?</command-message>", " ", text, flags=re.S)
    text = re.sub(r"<command-args>.*?</command-args>", " ", text, flags=re.S)
    text = re.sub(r"<local-command-stdout>.*?</local-command-stdout>", " ", text, flags=re.S)
    text = re.sub(r"<system-reminder>.*?</system-reminder>", " ", text, flags=re.S)
    text = re.sub(r"\[Request interrupted[^\]]*\]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------

def collect_claude_code(docs: list, titles: dict) -> None:
    root = HOME / ".claude" / "projects"
    if not root.exists():
        return
    for path in sorted(root.rglob("*.jsonl")):
        sid = f"claude_code:{path.stem}"
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            if rec.get("type") == "ai-title" and rec.get("aiTitle"):
                titles[sid] = rec["aiTitle"]
                continue
            if rec.get("type") not in ("user", "assistant") or rec.get("isMeta"):
                continue

            msg = rec.get("message")
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            texts: list[str] = []
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                for blk in content:
                    # §1.1: text only. no tool_use, no tool_result.
                    # `thinking` is stored empty on disk, so there is nothing to take.
                    if isinstance(blk, dict) and blk.get("type") == "text":
                        texts.append(blk.get("text") or "")

            for raw in texts:
                body = clean(raw)
                if len(body) < MIN_CHUNK_CHARS:
                    continue
                for chunk in split_text(body):
                    docs.append({
                        "session_id": sid,
                        "source": "claude_code",
                        "role": rec.get("type"),
                        "text": chunk,
                    })


def collect_codex(docs: list, titles: dict) -> None:
    root = HOME / ".codex"
    index = root / "session_index.jsonl"
    if index.exists():
        for line in index.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("id") and rec.get("thread_name"):
                titles[f"codex:{rec['id']}"] = rec["thread_name"]

    for path in sorted((root / "sessions").rglob("*.jsonl")) if (root / "sessions").exists() else []:
        sid = None
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            payload = rec.get("payload")
            if not isinstance(payload, dict):
                continue
            if rec.get("type") == "session_meta" and payload.get("id"):
                sid = f"codex:{payload['id']}"
                continue
            if sid is None:
                continue

            text = None
            if rec.get("type") == "response_item" and payload.get("role") in ("user", "assistant"):
                content = payload.get("content")
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    text = " ".join(
                        c.get("text", "") for c in content
                        if isinstance(c, dict) and "text" in c
                    )
            elif payload.get("type") in ("user_message", "agent_message"):
                text = payload.get("message") or payload.get("text")

            if not text:
                continue
            body = clean(text)
            if len(body) < MIN_CHUNK_CHARS:
                continue
            for chunk in split_text(body):
                docs.append({
                    "session_id": sid, "source": "codex",
                    "role": payload.get("role", payload.get("type")), "text": chunk,
                })


def collect_opencode(docs: list, titles: dict) -> None:
    root = HOME / ".local" / "share" / "opencode" / "storage"
    if not root.exists():
        return

    for path in sorted((root / "session").rglob("*.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(rec, dict) and rec.get("id") and rec.get("title"):
            titles[f"opencode:{rec['id']}"] = rec["title"]

    msg_role = {}
    for path in sorted((root / "message").rglob("*.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(rec, dict) and rec.get("id"):
            msg_role[rec["id"]] = (rec.get("role"), rec.get("sessionID"))

    for path in sorted((root / "part").rglob("*.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(rec, dict) or rec.get("type") != "text":
            continue
        role, sess = msg_role.get(rec.get("messageID", ""), (None, None))
        sess = sess or rec.get("sessionID")
        if not sess:
            continue
        body = clean(rec.get("text") or "")
        if len(body) < MIN_CHUNK_CHARS:
            continue
        for chunk in split_text(body):
            docs.append({
                "session_id": f"opencode:{sess}", "source": "opencode",
                "role": role or "unknown", "text": chunk,
            })


# --------------------------------------------------------------------------

def main() -> int:
    docs: list[dict] = []
    titles: dict[str, str] = {}

    collect_claude_code(docs, titles)
    collect_codex(docs, titles)
    collect_opencode(docs, titles)

    for i, d in enumerate(docs):
        d["doc_id"] = i

    per_session = Counter(d["session_id"] for d in docs)
    queries = [
        {"query": title, "target_session": sid,
         "source": sid.split(":", 1)[0],
         "lang": detect_lang(title),
         "n_docs": per_session[sid]}
        for sid, title in sorted(titles.items())
        if per_session.get(sid, 0) >= MIN_DOCS_PER_SESSION
    ]

    dropped = len(titles) - len(queries)
    lang_docs = Counter(detect_lang(d["text"]) for d in docs)

    payload = {
        "built_from": {
            "claude_code": sum(1 for d in docs if d["source"] == "claude_code"),
            "codex": sum(1 for d in docs if d["source"] == "codex"),
            "opencode": sum(1 for d in docs if d["source"] == "opencode"),
        },
        "n_docs": len(docs),
        "n_queries": len(queries),
        "n_sessions": len(per_session),
        "queries_dropped_thin_session": dropped,
        "query_lang": dict(Counter(q["lang"] for q in queries)),
        "doc_lang": dict(lang_docs),
        "docs": docs,
        "queries": queries,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    total_chars = sum(len(d["text"]) for d in docs)
    print(f"docs        {len(docs):>6}   ({total_chars/1e6:.2f} MB, ~{total_chars/4/1000:.0f}K tokens)")
    print(f"queries     {len(queries):>6}   (dropped {dropped} with <{MIN_DOCS_PER_SESSION} docs)")
    print(f"sessions    {len(per_session):>6}")
    print(f"by source   {payload['built_from']}")
    print(f"query lang  {payload['query_lang']}")
    print(f"doc lang    {payload['doc_lang']}")
    print(f"-> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
