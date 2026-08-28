"""Phase 3 — shape-probe a saved GitHub Copilot share page.

Why this needs its own probe rather than a branch in `probe_exports.py`:
every other source in this archive hands you a *file* — a ZIP or a JSON the provider
wrote on purpose. GitHub Copilot on github.com has no export at all. What it has is a
share link (`github.com/copilot/share/<uuid>`), and that link is **not public**: fetched
without a browser session it 302s to `/login`, and a `gh` OAuth token does not open it
either, because the route authenticates on session cookies rather than on a bearer token.

So the only thing that can reach the transcript is the browser that is already logged in.

**What Ctrl+S gets you is nothing, and that is the finding this probe exists to report.**
Measured on a real save (73 KB): the share route server-renders a mount point and no
more. `react-app.embeddedData.payload` is `{}`, the document holds 448 characters of
visible text — cookie banners and "Uh oh! There was an error while loading" — and the
messages are fetched afterwards, client-side, from `appPayload.apiURL`
(`api.individual.githubcopilot.com`, api version `2025-05-01`). A *Webpage, HTML only*
save re-requests that same empty shell, so it fails without looking like a failure.
`diagnose_shell` names that case rather than letting it read as "no candidates found".

Three containers can hold the real thing, and all three take the same candidate search:

  * **a HAR** from the network panel — the one capture that cannot pick the wrong
    request, since it holds every response and the search finds the transcript wherever
    it landed. Request headers are never read: a HAR taken against a logged-in session
    carries live bearer tokens in them.
  * **a single API response**, saved out of the network panel by hand.
  * **the rendered DOM** (Elements -> Copy outerHTML), which unlike Ctrl+S is what is
    actually on screen. Still supported, since GitHub could move the payload inline.

Run it against whatever you captured:

    python tools/probe_copilot_share.py                     # scan data/drops/
    python tools/probe_copilot_share.py ~/Downloads/x.har

Writes docs/formats/copilot_share.md and, when a transcript is found,
data/fixtures/exports/copilot_share.json.
"""

from __future__ import annotations

import html
import json
import re
import sys
from collections import Counter
from pathlib import Path

# Same cp1252 problem the CLI has: this report prints em-dashes and Slovene text, and a
# Windows console that cannot encode them would otherwise kill the probe mid-report.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

ROOT = Path(__file__).resolve().parent.parent
DROPS = ROOT / "data" / "drops"
FIXTURES = ROOT / "data" / "fixtures" / "exports"
REPORT = ROOT / "docs" / "formats" / "copilot_share.md"

MAX_STR = 160

# Attributes are captured, not skipped: `data-target` / `id` are how the adapter will
# find the right script among the dozen GitHub inlines on any page.
SCRIPT_RE = re.compile(
    rb'<script([^>]*\btype=["\']application/json["\'][^>]*)>(.*?)</script>',
    re.DOTALL | re.IGNORECASE)
ATTR_RE = re.compile(r'([\w:-]+)=["\']([^"\']*)["\']')

# Keys that mean "this dict is a conversation turn". Deliberately broad — the point of
# the probe is that we do not yet know which vocabulary this page uses.
ROLE_KEYS = {"role", "author", "sender", "speaker", "participant", "authorRole"}
TEXT_KEYS = {"text", "content", "body", "markdown", "message", "value", "parts"}


def truncate(obj, depth=0):
    if depth > 6:
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


def scripts(raw: bytes) -> list[dict]:
    """Every inline application/json block, decoded and parsed where possible.

    Browsers escape `<` and `&` when serialising a saved DOM, so the payload has to be
    HTML-unescaped before it is JSON. A block that still fails to parse is reported with
    its head rather than dropped — a truncated save is a finding, not a non-result.
    """
    found = []
    for match in SCRIPT_RE.finditer(raw):
        attrs = dict(ATTR_RE.findall(match.group(1).decode("utf-8", errors="replace")))
        body = html.unescape(match.group(2).decode("utf-8", errors="replace")).strip()
        entry = {
            "id": attrs.get("id"),
            "data_target": attrs.get("data-target"),
            "chars": len(body),
            "data": None,
            "error": None,
        }
        try:
            entry["data"] = json.loads(body)
        except json.JSONDecodeError as exc:
            entry["error"] = f"{exc.msg} at char {exc.pos}"
            entry["head"] = body[:400]
        found.append(entry)
    return found


def turn_arrays(node, path="$", depth=0, out=None) -> list[tuple[str, list]]:
    """Every list of turn-shaped dicts in the tree, in discovery order.

    "Turn-shaped" = a dict carrying both a role-ish key and a text-ish key. Matching on
    the pair rather than on either alone is what keeps this from returning every list of
    objects on the page.
    """
    out = [] if out is None else out
    if depth > 12:
        return out
    if isinstance(node, list):
        dicts = [x for x in node if isinstance(x, dict)]
        if len(dicts) >= 2 and all(
                (set(d) & ROLE_KEYS) and (set(d) & TEXT_KEYS) for d in dicts):
            out.append((path, dicts))
        for i, item in enumerate(node[:40]):
            turn_arrays(item, f"{path}[{i}]", depth + 1, out)
    elif isinstance(node, dict):
        for key, value in node.items():
            turn_arrays(value, f"{path}.{key}", depth + 1, out)
    return out


def describe_turns(turns: list[dict]) -> dict:
    roles = Counter()
    keys = Counter()
    chars = 0
    for turn in turns:
        keys.update(turn.keys())     # .update(dict) would read the VALUES as counts
        for key in ROLE_KEYS & set(turn):
            value = turn[key]
            roles[value if isinstance(value, str)
                  else json.dumps(value, default=str)[:40]] += 1
        for key in TEXT_KEYS & set(turn):
            value = turn[key]
            if isinstance(value, str):
                chars += len(value)
            elif isinstance(value, (list, dict)):
                chars += len(json.dumps(value, default=str))
    return {
        "turns": len(turns),
        "roles": dict(roles.most_common(8)),
        "keys": dict(keys.most_common(24)),
        "text_chars": chars,
    }


TAG_RE = re.compile(r"<(script|style)\b.*?</\1>", re.DOTALL | re.IGNORECASE)


def visible_text(raw: bytes) -> str:
    """Roughly what a reader would see. Crude on purpose — this measures *presence*.

    A rendered transcript runs to thousands of characters; the unrendered shell comes
    to a few hundred of cookie banners and "Uh oh! There was an error". Telling those
    two apart is the whole job, and it does not need a real HTML parser.
    """
    text = raw.decode("utf-8", errors="replace")
    body = text[text.find("<body"):] or text
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", TAG_RE.sub("", body))).strip()


def dom_evidence(raw: bytes) -> dict:
    """Does the saved HTML hold the transcript as markup rather than as JSON?

    Reported so a page with no embedded payload still produces an actionable answer:
    the adapter would then need to read the DOM, and the docs would have to insist on a
    save mode that keeps it.
    """
    text = raw.decode("utf-8", errors="replace")
    return {
        "html_chars": len(text),
        "visible_chars": len(visible_text(raw)),
        "markdown_body_divs": text.count("markdown-body"),
        "copilot_mentions": len(re.findall(r"copilot", text, re.IGNORECASE)),
        "looks_like_login": "Sign in to GitHub" in text,
    }


# Under ~1200 visible characters, a share page is chrome and error text with no
# conversation in it. Measured against the shell save: 448 characters.
SHELL_TEXT_MAX = 1200


def diagnose_shell(blocks: list[dict], dom: dict) -> dict | None:
    """Is this the app shell — the page *before* the transcript was fetched?

    The distinction the docs got wrong the first time. `/copilot/share/<uuid>` server-
    renders nothing but a mount point: `react-app.embeddedData.payload` is `{}`, and the
    messages are fetched client-side from the Copilot API named in `appPayload.apiURL`.
    So a "Webpage, HTML only" save — which re-requests the server HTML rather than
    serialising what is on screen — captures a page with no conversation in it, and
    reports no candidates for a reason that has nothing to do with the payload shape.
    """
    app = next((b["data"] for b in blocks
                if b.get("data_target") == "react-app.embeddedData"
                and isinstance(b.get("data"), dict)), None)
    if app is None:
        return None
    if app.get("payload") or dom["visible_chars"] > SHELL_TEXT_MAX:
        return None
    meta = app.get("appPayload") or {}
    return {
        "verdict": "app shell only — the transcript is not in this file",
        "payload_empty": app.get("payload") == {},
        "visible_chars": dom["visible_chars"],
        "fetched_at_runtime_from": meta.get("apiURL"),
        "api_version": meta.get("apiVersion"),
        "user": meta.get("currentUserLogin"),
    }


def har_blocks(doc: dict) -> list[dict]:
    """Every JSON response body in a HAR capture, as probeable blocks.

    A HAR is offered because it removes the one step that can go wrong by hand: picking
    the right request out of the network panel. The capture holds every response, so the
    candidate search finds the transcript wherever it landed, and the block is labelled
    with the URL that produced it — which is the thing the adapter needs to know.

    Request headers are never read. A HAR recorded against a logged-in session carries
    live bearer tokens in them, and this probe has no reason to touch those.
    """
    blocks = []
    for entry in doc.get("log", {}).get("entries", []):
        if not isinstance(entry, dict):
            continue
        url = (entry.get("request") or {}).get("url", "")
        content = (entry.get("response") or {}).get("content") or {}
        body = content.get("text")
        if not isinstance(body, str) or not body.strip():
            continue
        if content.get("encoding") == "base64":
            continue        # bytes, not a transcript
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            continue
        blocks.append({"id": url[:120], "data_target": None,
                       "chars": len(body), "data": data, "error": None})
    return blocks


def probe(path: Path) -> dict:
    raw = path.read_bytes()
    # A drop can also be the API response itself, or a whole HAR capture, taken from the
    # network panel — the only artefacts that hold the transcript when the page is a
    # shell. Both go down the same candidate search: one code path, three containers.
    if path.suffix.lower() in (".json", ".har"):
        try:
            doc = json.loads(raw.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as exc:
            return {"file": path.name, "bytes": len(raw), "json_scripts": [],
                    "candidates": [], "dom": {"looks_like_login": False,
                                              "visible_chars": 0,
                                              "markdown_body_divs": 0},
                    "error": f"not JSON: {exc.msg} at char {exc.pos}"}
        if isinstance(doc, dict) and isinstance(doc.get("log"), dict) \
                and "entries" in doc["log"]:
            blocks = har_blocks(doc)
        else:
            blocks = [{"id": path.name, "data_target": None, "chars": len(raw),
                       "data": doc, "error": None}]
    else:
        blocks = scripts(raw)
    report = {
        "file": path.name,
        "bytes": len(raw),
        "json_scripts": [
            {k: v for k, v in b.items() if k != "data"} | {
                "top_keys": (list(b["data"])[:20] if isinstance(b["data"], dict)
                             else f"<{type(b['data']).__name__}>")}
            for b in blocks
        ],
        "dom": dom_evidence(raw) if path.suffix.lower() not in (".json", ".har") else
               {"looks_like_login": False, "visible_chars": 0, "markdown_body_divs": 0},
        "candidates": [],
    }
    shell = diagnose_shell(blocks, report["dom"])
    if shell is not None:
        report["shell"] = shell

    best = None
    for block in blocks:
        for where, turns in turn_arrays(block["data"]):
            entry = {
                "script": block.get("data_target") or block.get("id") or "(anonymous)",
                "path": where,
                **describe_turns(turns),
            }
            report["candidates"].append(entry)
            if best is None or entry["text_chars"] > best[0]["text_chars"]:
                best = (entry, block["data"], turns)

    if best is not None:
        report["best"] = best[0]
        report["sample_turns"] = [truncate(t) for t in best[2][:3]]
        FIXTURES.mkdir(parents=True, exist_ok=True)
        (FIXTURES / "copilot_share.json").write_text(
            json.dumps(best[1], indent=2, default=str), encoding="utf-8")
        report["fixture"] = str((FIXTURES / "copilot_share.json").relative_to(ROOT))
    return report


def targets(argv: list[str]) -> list[Path]:
    if argv:
        out = []
        for arg in argv:
            path = Path(arg).expanduser()
            out.extend(sorted(path.glob("*.htm*")) if path.is_dir() else [path])
        return out
    if not DROPS.exists():
        return []
    # .json is scanned only when named explicitly: the drops folder is full of
    # other sources' exports, and a 14 MB T3 dump is not a Copilot share page.
    return sorted(p for p in DROPS.iterdir()
                  if p.suffix.lower() in (".html", ".htm", ".mhtml", ".har"))


def main() -> int:
    files = targets(sys.argv[1:])
    if not files:
        print(f"nothing to probe — save the share page into {DROPS}")
        return 1

    reports = []
    for path in files:
        if not path.exists():
            print(f"missing: {path}")
            continue
        rep = probe(path)
        reports.append(rep)
        print(f"\n=== {rep['file']}  ({rep['bytes']/1000:.0f} KB) ===")
        if rep.get("error"):
            print(f"  {rep['error']}")
            continue
        if rep["dom"]["looks_like_login"]:
            print("  this is GitHub's sign-in page, not the transcript —")
            print("  save it from a browser that is already logged in")
            continue
        if "shell" in rep:
            sh = rep["shell"]
            print(f"  {sh['verdict']}")
            print(f"  react-app payload is empty, {sh['visible_chars']} visible chars —")
            print(f"  the messages are fetched at runtime from "
                  f"{sh['fetched_at_runtime_from']} (api {sh['api_version']})")
            print("  capture the rendered DOM instead (DevTools -> Elements ->")
            print("  right-click <html> -> Copy outerHTML), or the API response JSON")
            continue
        for block in rep["json_scripts"]:
            name = block["data_target"] or block["id"] or "(anonymous)"
            note = f"  ERROR {block['error']}" if block["error"] else ""
            print(f"  script {name:<34} {block['chars']:>8} chars  "
                  f"{block['top_keys']}{note}")
        if not rep["candidates"]:
            print(f"  no turn-shaped arrays in any payload; "
                  f"markdown-body divs in DOM: {rep['dom']['markdown_body_divs']}")
        for cand in rep["candidates"]:
            print(f"  candidate {cand['path']}  turns={cand['turns']}  "
                  f"chars={cand['text_chars']}  roles={cand['roles']}")
        if "best" in rep:
            print(f"  -> fixture {rep['fixture']}")
            print(json.dumps(rep["sample_turns"], indent=2)[:2000])

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        "# GitHub Copilot share page — probed shape\n\n"
        "Generated by `tools/probe_copilot_share.py`.\n\n"
        "```json\n" + json.dumps(reports, indent=2, default=str) + "\n```\n",
        encoding="utf-8")
    print(f"\n-> {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
