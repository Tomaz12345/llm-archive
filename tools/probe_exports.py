"""Phase 0/3 — identify and shape-probe whatever lands in the drops folder.

Web exports arrive as ZIPs or bare JSON with no consistent naming, so files are
identified by sniffing their contents, never by filename. That is also how the real
ingest watcher will work.

    python tools/probe_exports.py                    # probe data/drops/
    python tools/probe_exports.py ~/Downloads        # probe somewhere else
    python tools/probe_exports.py --stage ~/Downloads  # copy recognised files into drops/

Writes docs/formats/exports.md and data/fixtures/exports/<kind>.json
"""

from __future__ import annotations

import json
import shutil
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DROPS = ROOT / "data" / "drops"
FIXTURES = ROOT / "data" / "fixtures" / "exports"
REPORT = ROOT / "docs" / "formats" / "exports.md"

sys.path.insert(0, str(ROOT))

# Google is the one export that is not JSON, and its rules — what counts as a Gemini
# activity log, where a cell ends, how a localised stamp is read — belong to the
# adapter. Imported rather than restated, so the probe cannot drift away from what
# ingest actually does.
from llm_archive.adapters.gemini import (           # noqa: E402
    GEMINI_URL, GeminiAdapter, _Reader,
)
from llm_archive.core.models import ParseStats      # noqa: E402

MAX_STR = 160


def truncate(obj, depth=0):
    if depth > 5:
        return "<deep>"
    if isinstance(obj, str):
        return obj if len(obj) <= MAX_STR else obj[:MAX_STR] + f"...<+{len(obj)-MAX_STR}>"
    if isinstance(obj, list):
        out = [truncate(x, depth + 1) for x in obj[:2]]
        if len(obj) > 2:
            out.append(f"<+{len(obj)-2} more>")
        return out
    if isinstance(obj, dict):
        return {k: truncate(v, depth + 1) for k, v in obj.items()}
    return obj


# --------------------------------------------------------------------------
# sniffing
# --------------------------------------------------------------------------

def sniff(data, name: str) -> str:
    """Identify an export by its structure. Filename is only a tiebreaker."""
    if isinstance(data, dict):
        v = str(data.get("version", ""))
        if v.startswith("orpg"):
            return "openrouter_chat"
        if "data_files" in data and "export_url" in json.dumps(data.get("data_files", ""))[:400]:
            return "claude_manifest"
        if "chat_messages" in data:
            return "claude_conversation"
        if "mapping" in data and "current_node" in data:
            return "chatgpt_conversation"
        if isinstance(data.get("threads"), list) and isinstance(data.get("messages"), list):
            return "t3chat_export"
        convs = data.get("conversations")
        if isinstance(convs, list) and "media_posts" in data:
            # Grok wraps each chat as {conversation, responses}; the sibling keys
            # (projects/tasks/media_posts) are there even when every one is empty.
            return "grok_export"
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            if "mapping" in first:
                # Both are arrays of node maps; only ChatGPT names the surviving leaf.
                return "chatgpt_export" if "current_node" in first else "deepseek_export"
            if "chat_messages" in first:
                return "claude_export"
            if "chatId" in first and "contentChunks" in first:
                # Le Chat exports the messages and nothing around them; `chatId` on
                # each record is the only conversation identity in the file.
                return "mistral_chat"
            if "uuid" in first and "name" in first and "created_at" in first:
                return "claude_projects"
    return "unknown"


def describe_openrouter(d) -> dict:
    msgs = d.get("messages") or {}
    items = d.get("items") or {}
    chars = d.get("characters") or {}
    roles = Counter()
    text_chars = 0
    for it in items.values():
        data = (it or {}).get("data") or {}
        roles[str(data.get("role"))] += 1
        for blk in data.get("content") or []:
            if isinstance(blk, dict) and blk.get("text"):
                text_chars += len(blk["text"])
    return {
        "title": d.get("title"),
        "schema_version": d.get("version"),
        "messages": len(msgs),
        "items": len(items),
        "characters": {c.get("id"): c.get("model") for c in chars.values()},
        "roles": dict(roles),
        "text_chars": text_chars,
        "artifacts": len(d.get("artifacts") or {}),
        "edited_msgs": sum(1 for m in msgs.values() if m.get("isEdited")),
        "retried_msgs": sum(1 for m in msgs.values() if m.get("isRetrying")),
        "text_path": "items[*].data.content[*].text",
        "role_path": "items[*].data.role  (characterId=='USER' for user)",
        "model_path": "characters[messages[*].characterId].model",
    }


def describe_t3chat(d) -> dict:
    """T3's bulk export: two flat arrays joined on threadId, no tree."""
    threads = [t for t in d.get("threads") or [] if isinstance(t, dict)]
    msgs = [m for m in d.get("messages") or [] if isinstance(m, dict)]
    per_thread = Counter(m.get("threadId") for m in msgs)
    part_types = Counter()
    tools = Counter()
    text_chars = 0
    providers = Counter()
    for m in msgs:
        for part in m.get("parts") or []:
            if not isinstance(part, dict):
                continue
            part_types[str(part.get("type"))] += 1
            if part.get("type") == "tool_call":
                tools[str(part.get("toolName"))] += 1
            text_chars += len(part.get("text") or part.get("reasoning") or "")
        providers.update(k for k in (m.get("providerMetadata") or {}))
    return {
        "schema_version": d.get("version"),
        "threads": len(threads),
        "messages": len(msgs),
        "roles": dict(Counter(m.get("role") for m in msgs)),
        "statuses": dict(Counter(m.get("status") for m in msgs)),
        "models": dict(Counter(m.get("model") for m in msgs).most_common(8)),
        "part_types": dict(part_types),
        "tools": dict(tools),
        "usage_blocks": dict(providers),
        "text_chars": text_chars,
        "threads_without_messages": sum(1 for t in threads
                                        if not per_thread.get(t.get("threadId"))),
        "msgs_without_thread": sum(n for tid, n in per_thread.items()
                                   if tid not in {t.get("threadId") for t in threads}),
        "attachment_refs": sum(len(m.get("attachmentIds") or []) for m in msgs),
        "parent_pointers": 0,      # none exist: this format is flat, see PLAN §2.2
        "text_path": "messages[*].parts[*].text  (fallback messages[*].content)",
        "join_path": "messages[*].threadId -> threads[*].threadId",
        "model_path": "messages[*].model  (threads[*].model is only the last pick)",
    }


def describe_deepseek(d) -> dict:
    """DeepSeek's account export: an array of ChatGPT-shaped node maps, no current_node."""
    convs = [c for c in d if isinstance(c, dict)]
    frags = Counter()
    roles = Counter()
    branched = inverted = search_hits = text_chars = 0
    models = Counter()
    for conv in convs:
        mapping = {k: v for k, v in (conv.get("mapping") or {}).items()
                   if isinstance(v, dict)}
        kids = Counter()
        stamp = {}
        for nid, node in mapping.items():
            kids[str(node.get("parent"))] += 1
            msg = node.get("message")
            if not isinstance(msg, dict):
                continue
            models[str(msg.get("model"))] += 1
            stamp[nid] = str(msg.get("inserted_at") or "")
            types = [str(f.get("type")) for f in msg.get("fragments") or []
                     if isinstance(f, dict)]
            frags.update(types)
            roles["user" if "REQUEST" in types else "assistant"] += 1
            for frag in msg.get("fragments") or []:
                if not isinstance(frag, dict):
                    continue
                text_chars += len(frag.get("content") or "")
                search_hits += len(frag.get("results") or [])
        branched += any(n > 1 for n in kids.values())
        # An answer stamped before the prompt that produced it — see adapters/deepseek.py
        inverted += sum(1 for nid, node in mapping.items()
                        if (par := str(node.get("parent"))) in stamp and nid in stamp
                        and stamp[nid] < stamp[par])
    return {
        "conversations": len(convs),
        "nodes": sum(len(c.get("mapping") or {}) for c in convs),
        "roles": dict(roles),
        "fragment_types": dict(frags),
        "models": dict(models.most_common(8)),
        "branched_conversations": branched,
        "backwards_timestamps": inverted,
        "search_hits": search_hits,
        "text_chars": text_chars,
        "current_node": False,      # the one ChatGPT field this format omits
        "token_counts": False,      # no usage on any record
        "text_path": "mapping[*].message.fragments[*].content",
        "role_path": "fragment type: REQUEST=user, RESPONSE/SEARCH=assistant",
        "model_path": "mapping[*].message.model  (set on prompts too; selected, not speaker)",
    }


def describe_mistral(d) -> dict:
    """Le Chat's per-chat export: a bare array of message records, no chat object."""
    msgs = [m for m in d if isinstance(m, dict)]
    roles = Counter(str(m.get("role")) for m in msgs)
    chunk_types = Counter()
    contexts = Counter()
    text_chars = duplicated = reasoning_chars = tok_out = 0
    for m in msgs:
        text_chars += len(m.get("content") or "")
        chunks = m.get("contentChunks")
        for chunk in chunks or []:
            if not isinstance(chunk, dict):
                continue
            chunk_types[str(chunk.get("type"))] += 1
            ctx = chunk.get("_context") if isinstance(chunk.get("_context"), dict) else {}
            contexts[str(ctx.get("type") or "-")] += 1
            if str(ctx.get("type")) == "reasoning":
                reasoning_chars += len(chunk.get("text") or "")
            # The trap: the final chunk repeats `content` byte for byte.
            elif (chunk.get("text") or "") == (m.get("content") or ""):
                duplicated += 1
        gen = (((m.get("context") or {}).get("completionTiming") or {})
               .get("generationStats") or {})
        tok_out += gen.get("outputTokens") or 0
    return {
        "chats": len({str(m.get("chatId")) for m in msgs}),
        "messages": len(msgs),
        "roles": dict(roles),
        "chunk_types": dict(chunk_types),
        "chunk_contexts": dict(contexts),
        "answers_duplicated_in_content": duplicated,
        "reasoning_chars": reasoning_chars,
        "text_chars": text_chars,
        "versions": dict(Counter(m.get("version") for m in msgs)),
        "tok_out": tok_out,
        "models_recorded": 0,          # nothing anywhere names the model
        "canvas_entries": sum(len(m.get("canvas") or []) for m in msgs),
        "files": sum(len(m.get("files") or []) for m in msgs),
    }


def describe_grok(d) -> dict:
    """Grok's account export: {conversation, responses} pairs with a named leaf."""
    entries = [c for c in d.get("conversations") or [] if isinstance(c, dict)]
    senders, tools, tags, models, pickers = (Counter() for _ in range(5))
    text_chars = hits = steps = no_leaf = branched = 0
    results = 0
    for entry in entries:
        conv = entry.get("conversation") or {}
        rows = [(w.get("response") or {}) for w in entry.get("responses") or []
                if isinstance(w, dict)]
        by_id = {str(r.get("_id")): r for r in rows if r.get("_id")}
        no_leaf += str(conv.get("leaf_response_id")) not in by_id
        kids = Counter(str(r.get("parent_response_id")) for r in rows)
        branched += any(n > 1 for n in kids.values())
        for r in rows:
            senders[str(r.get("sender"))] += 1
            text_chars += len(r.get("message") or "")
            request = ((r.get("metadata") or {}).get("request_metadata") or {})
            models[str(request.get("resolved_model") or "")] += 1
            pickers[str(r.get("model") or "")] += 1
            hits += len(r.get("web_search_results") or [])
            for step in r.get("steps") or []:
                if not isinstance(step, dict):
                    continue
                steps += 1
                tags.update(str(t) for t in step.get("tag_order") or [])
                results += len(step.get("tool_usage_results") or [])
                for card in step.get("tool_usage_cards") or []:
                    tools.update(str(k) for k in (card.get("tool") or {}))
    return {
        "conversations": len(entries),
        "responses": sum(senders.values()),
        "senders": dict(senders),
        "models": dict(models.most_common(8)),
        "pickers": dict(pickers.most_common(8)),
        "steps": steps,
        "step_tags": dict(tags),
        "tool_calls": dict(tools),
        "tool_results": results,
        "calls_without_result": sum(tools.values()) - results,
        "search_hits_aggregate": hits,
        "branched_conversations": branched,
        "conversations_without_leaf_in_export": no_leaf,
        "text_chars": text_chars,
        "side_collections": {k: len(d.get(k) or [])
                             for k in ("projects", "tasks", "media_posts")},
        "token_counts": False,      # no usage on any record
        "text_path": "conversations[*].responses[*].response.message",
        "role_path": "response.sender: human=user, assistant=assistant",
        "model_path": "response.metadata.request_metadata.resolved_model "
                      "(response.model is the UI picker)",
        "join_path": "response.parent_response_id -> response._id; "
                     "conversation.leaf_response_id names the surviving leaf",
    }


def describe_gemini(document: str) -> dict:
    """Google Takeout's My Activity log: not conversations, stamped events — PLAN §2.4."""
    adapter = GeminiAdapter()
    stats = ParseStats()
    cells = GeminiAdapter._split(document)
    parsed = [c for c in (adapter._cell(chunk, i, stats)
                          for i, chunk in enumerate(cells)) if c is not None]
    threads = Counter(c.conv_id for c in parsed if c.conv_id)
    labels = Counter(c.label for c in parsed)
    stamps = sorted(c.at for c in parsed if c.at)
    return {
        "cells": len(cells),
        "threads": len(threads),
        "canvases": sum(1 for c in parsed if not c.conv_id and c.reply),
        "prompts": sum(1 for c in parsed if c.asked),
        "markers_without_content": sum(1 for c in parsed
                                       if not c.asked and not c.reply),
        "attachments": sum(len(c.attachments) for c in parsed),
        "activity_labels": dict(labels.most_common(8)),
        "biggest_thread": max(threads.values(), default=0),
        "timezones": dict(Counter(c.tz for c in parsed)),
        "span": [datetime.fromtimestamp(t / 1000, timezone.utc).date().isoformat()
                 for t in (stamps[:1] + stamps[-1:])],
        "text_chars": sum(len(c.prompt) + len(c.reply) for c in parsed),
        "unparsed": dict(stats.unknown_types),
        "roles": False,             # nowhere in the file — inferred from the timestamp
        "model": False,             # never recorded, not even the family
        "token_counts": False,      # no usage on any record
        "text_path": "outer-cell > content-cell.body-1, split on the localised stamp",
        "role_path": "before the stamp = you, after it = Gemini; NBSP+colon marks a prompt",
        "join_path": "caption > gemini.google.com/app/<id>  (absent on canvases)",
    }


def describe_generic(d) -> dict:
    if isinstance(d, list):
        return {"top_level": "array", "n": len(d),
                "item_keys": sorted(d[0].keys()) if d and isinstance(d[0], dict) else None}
    return {"top_level": "object", "keys": sorted(d.keys())}


# --------------------------------------------------------------------------

def load_any(path: Path):
    """Yield (label, parsed_json) from a .json file or every .json inside a .zip."""
    if path.suffix.lower() == ".json":
        try:
            yield path.name, json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  ! {path.name}: {type(exc).__name__}")
        return
    if path.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(path) as zf:
                for member in zf.namelist():
                    if not member.lower().endswith(".json"):
                        continue
                    try:
                        with zf.open(member) as fh:
                            yield f"{path.name}::{member}", json.loads(
                                fh.read().decode("utf-8", errors="replace"))
                    except (json.JSONDecodeError, KeyError):
                        print(f"  ! {path.name}::{member}: bad json")
        except zipfile.BadZipFile:
            print(f"  ! {path.name}: not a zip")


def load_html(path: Path):
    """Yield (label, html) for every Gemini activity log in a ZIP, folder or file."""
    if path.suffix.lower() not in (".zip", ".html") and not path.is_dir():
        return
    reader = None
    try:
        reader = _Reader(path)
        for member in reader.documents():
            document = reader.text(member)
            if GeminiAdapter._is_gemini(document[:1 << 20]):
                yield f"{path.name}::{member.rsplit('/', 1)[-1]}", document
    except (OSError, zipfile.BadZipFile, RuntimeError):
        print(f"  ! {path.name}: unreadable")
    finally:
        if reader is not None:
            reader.close()


def main() -> int:
    argv = [a for a in sys.argv[1:] if not a.startswith("-")]
    stage = "--stage" in sys.argv
    search = Path(argv[0]).expanduser() if argv else DROPS
    DROPS.mkdir(parents=True, exist_ok=True)
    FIXTURES.mkdir(parents=True, exist_ok=True)

    if not search.exists():
        print(f"nothing at {search}")
        return 1

    candidates = [p for p in sorted(search.iterdir())
                  if p.is_file() and p.suffix.lower() in (".json", ".zip", ".html")]
    print(f"scanning {search}  ({len(candidates)} json/zip/html files)\n")

    found = []
    for path in candidates:
        for label, data in load_any(path):
            kind = sniff(data, label)
            if kind == "unknown":
                continue
            describe = {"openrouter_chat": describe_openrouter,
                        "t3chat_export": describe_t3chat,
                        "deepseek_export": describe_deepseek,
                        "grok_export": describe_grok,
                        "mistral_chat": describe_mistral}.get(kind, describe_generic)
            detail = describe(data)
            found.append((kind, label, path, detail, data))
            print(f"  [{kind:22s}] {label}")

        # Takeout ships HTML, so it never reaches the JSON sniffer above.
        for label, document in load_html(path):
            found.append(("gemini_activity", label, path,
                          describe_gemini(document), document))
            print(f"  [{'gemini_activity':22s}] {label}")

    if not found:
        print("  no recognised exports found")

    # group + fixtures
    by_kind = Counter(k for k, *_ in found)
    for kind in by_kind:
        sample = next(d for k, _, _, _, d in found if k == kind)
        if isinstance(sample, str):
            # An activity log: the fixture is the markup itself, first two cells of it,
            # which is what a parser regression actually needs to see.
            cells = GeminiAdapter._split(sample)[:2]
            (FIXTURES / f"{kind}.html").write_text(
                GEMINI_URL.sub("https://gemini.google.com/app/<id>", "".join(cells)),
                encoding="utf-8")
            continue
        (FIXTURES / f"{kind}.json").write_text(
            json.dumps(truncate(sample), indent=2, ensure_ascii=False), encoding="utf-8")

    if stage:
        print("\nstaging into data/drops/")
        for kind, label, path, _, _ in found:
            if kind == "claude_manifest":
                continue  # the manifest is not data
            dest = DROPS / path.name
            if not dest.exists():
                shutil.copy2(path, dest)
                print(f"  + {path.name}")

    # report
    lines = [
        "# Export probe",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} "
        f"from `{search}` by `tools/probe_exports.py`.",
        "",
        "Files are identified by **sniffing their structure**, never by filename — that is "
        "how the ingest watcher will work too.",
        "",
        "| kind | files |",
        "|---|---:|",
    ]
    for kind, n in by_kind.most_common():
        lines.append(f"| `{kind}` | {n} |")
    lines.append("")

    for kind in by_kind:
        lines += [f"## `{kind}`", ""]
        for k, label, _, detail, _ in found:
            if k != kind:
                continue
            lines.append(f"**{label}**")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(detail, indent=2, ensure_ascii=False)[:2000])
            lines.append("```")
            lines.append("")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n-> {REPORT.relative_to(ROOT)}")
    print(f"-> {FIXTURES.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
