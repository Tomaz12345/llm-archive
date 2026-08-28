"""llma — command line for the session archive."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import typer

from .core import db, idb, ingest
from .core.blobs import BlobStore

# Windows hands Python a cp1252 stdout, which cannot encode č/š/ž — so printing a
# Slovene snippet killed `search` with a UnicodeEncodeError mid-result. Half this
# archive is Slovene, so this is not an edge case. `errors="replace"` keeps a terminal
# that genuinely cannot render a glyph from taking the command down with it.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass        # pytest's capture object and friends are not reconfigurable

app = typer.Typer(add_completion=False, help="Local archive of LLM sessions.")


def _open(data_dir: Path | None):
    db_path, blob_dir = ingest.default_paths(data_dir)
    return db.connect(db_path), BlobStore(blob_dir), db_path


def _report(res, echo=typer.echo) -> None:
    """The per-adapter result block, shared by `ingest` and `add`."""
    line = (f"  files {res.files}  new {res.new}  updated {res.updated}  "
            f"skipped {res.skipped}")
    # Only shown when they happened: a run of zeroes on every source teaches you to
    # skim past the line, and these three are the ones worth noticing.
    if res.appended:
        line += f"  appended {res.appended}"
    if res.retained:
        line += f"  retained {res.retained}"
    if res.stale:
        line += f"  stale {res.stale}"
    echo(line)
    echo(f"  messages {res.messages}  parts {res.parts}  "
         f"orphaned {res.orphaned}  blobs {res.blobs} "
         f"({res.blob_bytes/1e6:.1f} MB)")
    echo(f"  {res.seconds:.1f}s")
    if res.drops_archived:
        echo(f"  archived {res.drops_archived} spent drop(s) to drops/_archive/")
    if res.errors:
        echo(f"  errors: {res.errors}")
    if res.redacted:
        echo(f"  redacted {res.redacted} secret(s) before indexing")
    if res.unknown_types:
        top = list(res.unknown_types.items())[:6]
        echo(f"  unhandled record types (counted, not fatal): {dict(top)}")


@app.command("add")
def add_cmd(
    paths: list[Path] = typer.Argument(..., help="files, folders, or globs to take in"),
    no_ingest: bool = typer.Option(False, "--no-ingest",
                                   help="file the exports but do not read them yet"),
    show_all: bool = typer.Option(False, "--all",
                                  help="also list the files that were not exports"),
    data_dir: Path = typer.Option(None, "--data-dir", help="override data/ location"),
) -> None:
    """Take export files into the archive, working out what they are.

    Point it at anything — a downloaded ZIP, a folder full of them, your whole
    Downloads directory. Each file is identified by its contents rather than its name
    (Grok's export is a bare uuid; Mistral's is a timestamp), copied into data/drops/,
    and then read. The originals are left exactly where they were.

        llma add ~/Downloads/conversations-000.zip
        llma add ~/Downloads              # scans, claims only what is an export
        llma add --no-ingest export.zip   # file it now, read it on the next sync

    Re-adding a file you already have is a no-op: identity is the sha256, so the same
    export downloaded twice under two names is recognised as one.
    """
    from .core import intake

    con, blobs, db_path = _open(data_dir)
    drops = ingest.drops_dir(data_dir)

    taken = intake.take_all(paths, con, drops)
    if not taken:
        typer.echo("nothing to look at")
        raise typer.Exit(1)

    # A Downloads folder is 400 files of which 10 are exports. Listing every rejection
    # buries the ten lines you are actually reading the output for, so the misses are
    # collapsed to a count unless you ask for them.
    interesting = [t for t in taken if t.kind or t.action == "failed"] \
        if not show_all else taken
    passed_over = len(taken) - len(interesting)

    if interesting:
        width = min(max(len(t.source.name) for t in interesting), 52)
        for t in interesting:
            name = t.source.name if len(t.source.name) <= width \
                else t.source.name[:width - 1] + "…"
            typer.echo(f"  {name:<{width}}  {t.kind or '—':<12} {t.action}"
                       f"{'  (' + t.detail + ')' if t.detail else ''}")

    kinds = sorted({t.kind for t in taken if t.ok and t.kind})
    added = sum(1 for t in taken if t.ok)
    failed = [t for t in taken if t.action == "failed"]

    summary = f"\n{added} added, {sum(1 for t in taken if t.action == 'held')} already held"
    if passed_over:
        summary += f", {passed_over} not exports"
    if failed:
        summary += f", {len(failed)} failed"
    typer.echo(summary)
    if passed_over and not show_all:
        typer.echo("  (--all lists them)")

    if no_ingest or not kinds:
        if kinds:
            typer.echo("not ingested (--no-ingest); run `llma sync` when ready")
        elif not added:
            typer.echo("nothing new to ingest")
        raise typer.Exit(0)

    for kind in kinds:
        adapters = ingest.build_adapters(blobs, kind, drops=drops,
                                         browser=idb.is_enabled(con))
        for adapter in adapters:
            typer.echo(f"\n=== {adapter.label} ===")
            _report(ingest.run(adapter, con, blobs))

    typer.echo(f"\n-> {db_path}")
    typer.echo("run `llma index` (or `llma sync`) to make the new sessions searchable")


@app.command("browser")
def browser_cmd(
    enable: bool = typer.Option(False, "--enable",
                                help="read the browser store on every ingest"),
    disable: bool = typer.Option(False, "--disable"),
    data_dir: Path = typer.Option(None, "--data-dir"),
) -> None:
    """Read OpenRouter's chats from your browser instead of exporting them one by one.

    OpenRouter is local-first and has no bulk export: capturing 50 chats means 50
    separate "Export Chat" clicks. The chats are already on this disk — the site keeps
    them in its own IndexedDB store — so this reads them from there.

    Off by default, and it stays off until you say otherwise, because it opens a
    browser profile directory. Only the store belonging to openrouter.ai is ever
    touched, it is copied before being read (a running browser holds a lock on the
    live one), and nothing else in the profile is looked at.

    Chats read this way merge with ones you exported by hand — both routes key a
    session on the same root message id — so turning this on does not duplicate
    anything you already have.

    With no flags, reports what is on this machine.
    """
    con, _, _ = _open(data_dir)
    if enable and disable:
        typer.echo("--enable and --disable are opposites; pick one")
        raise typer.Exit(1)
    if enable or disable:
        idb.set_enabled(con, enable)
        typer.echo(f"browser store reading is {'ON' if enable else 'OFF'}"
                   f"{' for every ingest from now on' if enable else ''}")

    from .adapters.openrouter import browser_stores

    typer.echo(f"setting    {'on' if idb.is_enabled(con) else 'off'}")
    stores = browser_stores()
    if not stores:
        typer.echo("stores     none found")
        typer.echo("           OpenRouter keeps nothing locally for this account, you "
                   "use a browser\n           this cannot read (only Firefox is "
                   "supported), or the profile is elsewhere.")
        return
    for store in stores:
        typer.echo(f"store      {store.name}")
    if not idb.is_enabled(con):
        typer.echo("\nrun `llma browser --enable` to read them on every ingest")


@app.command("ingest")
def ingest_cmd(
    source: str = typer.Option(None, "--source", "-s", help="only this adapter"),
    root: Path = typer.Option(None, "--root",
                              help="a store copied from another machine "
                                   "(requires --source and --host)"),
    host: str = typer.Option(None, "--host",
                             help="machine the data came from; defaults to this one"),
    force: bool = typer.Option(False, "--force",
                               help="re-parse even if the source bytes are unchanged "
                                    "(use after an adapter change)"),
    browser: bool = typer.Option(False, "--browser",
                                 help="also read OpenRouter's own browser store, "
                                      "just this once (see: llma browser)"),
    data_dir: Path = typer.Option(None, "--data-dir", help="override data/ location"),
) -> None:
    """Read session stores into the archive. Safe to re-run.

    To pull in another machine's sessions, copy its store here and name the machine:

        llma ingest --source claude_code --root D:/from-laptop/.claude --host laptop
        llma ingest --source codex       --root D:/from-laptop/.codex  --host laptop

    Nothing in these formats records a hostname, so --host is the only thing keeping
    three machines' sessions apart in the statistics.
    """
    con, blobs, db_path = _open(data_dir)
    if root is not None and not root.exists():
        typer.echo(f"--root does not exist: {root}")
        raise typer.Exit(1)
    if root is not None and not host:
        typer.echo("--root needs --host so the sessions can be told apart later "
                   "(e.g. --host laptop)")
        raise typer.Exit(1)
    try:
        adapters = ingest.build_adapters(blobs, source, root=root, host=host,
                                         drops=ingest.drops_dir(data_dir),
                                         browser=browser or idb.is_enabled(con))
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1)
    if not adapters:
        typer.echo(f"no adapter named {source!r}")
        raise typer.Exit(1)
    if root is not None:
        typer.echo(f"importing {source} from {root}  (host={host})")

    for adapter in adapters:
        typer.echo(f"\n=== {adapter.label} ===")
        _report(ingest.run(adapter, con, blobs, force=force))
    typer.echo(f"\n-> {db_path}")


@app.command()
def stats(data_dir: Path = typer.Option(None, "--data-dir")) -> None:
    """Summarise what is in the archive."""
    con, _, _ = _open(data_dir)
    q = con.execute

    total = q("SELECT COUNT(*) n FROM session").fetchone()["n"]
    if not total:
        typer.echo("archive is empty — run: llma ingest")
        raise typer.Exit()

    # `turns` before `msgs`: only the first is comparable between these rows --
    # an agent files each tool call and result as its own message, a chat app
    # folds them into the reply.
    typer.echo("BY SOURCE")
    for r in q("""SELECT s.label, COUNT(*) sessions, SUM(x.msg_count) msgs,
                         SUM(x.turn_count) turns,
                         SUM(x.tok_in) ti, SUM(x.tok_out) to_,
                         SUM(x.tok_cache_read) cr, SUM(x.tok_cache_write) cw
                  FROM session x JOIN source s ON s.id=x.source_id
                  GROUP BY s.id ORDER BY sessions DESC"""):
        typer.echo(f"  {r['label']:<16} {r['sessions']:>4} sessions  "
                   f"{r['turns'] or 0:>6} turns  {r['msgs'] or 0:>6} msgs")
        typer.echo(f"    tokens  in {(r['ti'] or 0)/1e6:>7.2f}M   "
                   f"out {(r['to_'] or 0)/1e6:>7.2f}M   "
                   f"cache-read {(r['cr'] or 0)/1e6:>9.1f}M   "
                   f"cache-write {(r['cw'] or 0)/1e6:>7.1f}M")

    sub = q("SELECT COUNT(*) n FROM session WHERE parent_session_id IS NOT NULL").fetchone()
    if sub["n"]:
        typer.echo(f"    {sub['n']} subagent session(s) linked to a parent")

    # One store, several assistants: the VS Code panel is shared by Copilot Chat and
    # any other chat extension, so the source lines above hide who actually answered.
    # GROUP BY the expression, not the output alias: `label` would bind to source.label
    # instead and collapse every assistant into one row per source.
    people = q("""SELECT COALESCE(json_extract(x.meta,'$.participant_label'),
                                 json_extract(x.meta,'$.participant')) who,
                         s.label AS source, COUNT(*) n, SUM(x.msg_count) msgs
                  FROM session x JOIN source s ON s.id=x.source_id
                  WHERE json_extract(x.meta,'$.participant') IS NOT NULL
                  GROUP BY json_extract(x.meta,'$.participant'), s.id
                  ORDER BY n DESC""").fetchall()
    if people:
        typer.echo("\nBY ASSISTANT")
        for r in people:
            typer.echo(f"  {r['who']:<22} {r['n']:>4} sessions  "
                       f"{r['msgs'] or 0:>6} msgs   ({r['source']})")

    hosts = q("""SELECT COALESCE(host,'(web)') h, COUNT(*) n, SUM(msg_count) m,
                        SUM(turn_count) t
                 FROM session GROUP BY host ORDER BY n DESC""").fetchall()
    if len(hosts) > 1:
        typer.echo("\nBY MACHINE")
        for r in hosts:
            typer.echo(f"  {r['h']:<22} {r['n']:>4} sessions  "
                       f"{r['t'] or 0:>6} turns  {r['m'] or 0:>6} msgs")

    typer.echo("\nBY WORKSPACE")
    for r in q("""SELECT COALESCE(w.label,'(none)') label, w.key,
                         COUNT(*) n, SUM(x.msg_count) msgs
                  FROM session x LEFT JOIN workspace w ON w.id=x.workspace_id
                  GROUP BY x.workspace_id ORDER BY n DESC LIMIT 12"""):
        typer.echo(f"  {r['label']:<28} {r['n']:>3} sessions  {r['msgs'] or 0:>6} msgs")

    typer.echo("\nCONTENT")
    typer.echo(f"  {'kind':<14} {'parts':>6} {'original':>10} {'embedded':>10}")
    for r in q("""SELECT kind, COUNT(*) n, SUM(bytes) b,
                         SUM(CASE WHEN embed_eligible=1 THEN LENGTH(text) ELSE 0 END) et
                  FROM part GROUP BY kind ORDER BY b DESC"""):
        emb = f"{(r['et'] or 0)/1e6:>9.2f}M" if r["et"] else "         —"
        typer.echo(f"  {r['kind']:<14} {r['n']:>6} {(r['b'] or 0)/1e6:>9.1f}M {emb}")

    tot = q("""SELECT SUM(bytes) b,
                      SUM(CASE WHEN embed_eligible=1 THEN LENGTH(text) ELSE 0 END) et
               FROM part""").fetchone()
    # The §1.1 ratio, measured rather than assumed: how little of the archive is
    # actually worth embedding.
    typer.echo(f"  {'':<14} {'':>6} {'-'*10} {'-'*10}")
    typer.echo(f"  {'TOTAL':<14} {'':>6} {(tot['b'] or 0)/1e6:>9.1f}M "
               f"{(tot['et'] or 0)/1e6:>9.2f}M")
    if tot["b"]:
        typer.echo(f"  embed corpus is {100*(tot['et'] or 0)/tot['b']:.1f}% of stored bytes "
                   f"(~{(tot['et'] or 0)/4/1000:.0f}K tokens)")

    row = q("""SELECT COUNT(*) total,
                      SUM(on_active_path) active,
                      SUM(is_sidechain) side FROM message""").fetchone()
    typer.echo(f"\nMESSAGES  {row['total']} total  "
               f"{row['active']} on active path  "
               f"{row['total']-row['active']} abandoned  "
               f"{row['side']} sidechain")

    b = q("SELECT COUNT(*) n, SUM(bytes) b FROM blob").fetchone()
    typer.echo(f"BLOBS     {b['n'] or 0} files  {(b['b'] or 0)/1e6:.1f} MB")

    typer.echo("\nMODELS")
    for r in q("""SELECT model, COUNT(*) n FROM message
                  WHERE model IS NOT NULL GROUP BY model ORDER BY n DESC LIMIT 8"""):
        typer.echo(f"  {r['model']:<34} {r['n']:>6}")


@app.command("index")
def index_cmd(
    no_vectors: bool = typer.Option(False, "--no-vectors",
                                    help="keyword index only (seconds, not minutes)"),
    data_dir: Path = typer.Option(None, "--data-dir"),
) -> None:
    """Build the search indexes. Re-run after ingesting."""
    from .search import index as search_index

    con, _, db_path = _open(data_dir)
    vectors_dir = (data_dir or db_path.parent) / "vectors"

    state = {"last": -1}

    def progress(done: int, total: int) -> None:
        pct = int(done * 100 / total)
        if pct >= state["last"] + 5:
            state["last"] = pct
            typer.echo(f"  embedding {done}/{total}  ({pct}%)")

    if not no_vectors:
        typer.echo("building keyword index, then embedding "
                   "(first run takes a while on CPU)...")
    res = search_index.build(con, vectors_dir, with_vectors=not no_vectors,
                             progress=None if no_vectors else progress)

    typer.echo(f"\n  FTS rows   {res.fts_rows}")
    typer.echo(f"  chunks     {res.chunks}  ({res.chars/1e6:.2f} MB)")
    if res.skipped_vectors:
        typer.echo("  vectors    skipped — keyword search only")
    else:
        typer.echo(f"  vectors    {res.vectors} x 384  [{res.model_tag}]")
    typer.echo(f"  {res.seconds:.1f}s")
    for warning in res.warnings:
        typer.echo(f"  ! {warning}")


@app.command("search")
def search_cmd(
    query: str = typer.Argument(..., help="what you are looking for"),
    limit: int = typer.Option(10, "--limit", "-n"),
    source: list[str] = typer.Option(None, "--source", "-s", help="repeatable"),
    participant: str = typer.Option(None, "--participant", "-p",
                                    help="assistant inside a shared panel, "
                                         "e.g. copilot (see: llma stats)"),
    workspace: str = typer.Option(None, "--workspace", "-w"),
    host: str = typer.Option(None, "--host"),
    since: str = typer.Option(None, "--since", help="YYYY-MM-DD"),
    until: str = typer.Option(None, "--until", help="YYYY-MM-DD"),
    mode: str = typer.Option("hybrid", "--mode",
                             help="hybrid | keyword | semantic"),
    abandoned: bool = typer.Option(False, "--abandoned",
                                   help="include abandoned branches"),
    data_dir: Path = typer.Option(None, "--data-dir"),
) -> None:
    """Search the archive."""
    from .search.hybrid import Filters
    from .search.hybrid import search as run_search

    con, _, db_path = _open(data_dir)
    vectors_dir = (data_dir or db_path.parent) / "vectors"

    def as_ms(value: str | None) -> int | None:
        if not value:
            return None
        return int(datetime.strptime(value, "%Y-%m-%d")
                   .replace(tzinfo=timezone.utc).timestamp() * 1000)

    filters = Filters(sources=tuple(source or ()), participant=participant,
                      workspace=workspace, host=host,
                      since=as_ms(since), until=as_ms(until),
                      include_abandoned=abandoned)
    hits = run_search(con, vectors_dir, query, limit=limit, filters=filters, mode=mode)

    if not hits:
        typer.echo("no matches")
        raise typer.Exit()

    for i, hit in enumerate(hits, 1):
        when = (datetime.fromtimestamp(hit.started_at / 1000, timezone.utc)
                .strftime("%Y-%m-%d") if hit.started_at else "?")
        where = f" · {hit.workspace}" if hit.workspace else ""
        machine = f" · {hit.host}" if hit.host else ""
        typer.echo(f"\n{i:>2}. {hit.title or '(untitled)'}")
        typer.echo(f"    {when} · {hit.source}{where}{machine}"
                   f" · {hit.matched_by} · {hit.score:.4f}  [#{hit.session_id}]")
        for snip in hit.snippets[:2]:
            text = " ".join((snip.text or "").split())
            if text:
                typer.echo(f"    {snip.role[:9]:<9} {text[:150]}")


@app.command()
def show(
    session_id: int = typer.Argument(..., help="session id from search results"),
    tools: bool = typer.Option(False, "--tools", help="include tool calls"),
    abandoned: bool = typer.Option(False, "--abandoned"),
    data_dir: Path = typer.Option(None, "--data-dir"),
) -> None:
    """Print one session as a readable transcript."""
    con, _, _ = _open(data_dir)
    meta = con.execute("""
        SELECT s.title, s.started_at, s.host, s.raw_path, src.kind AS source,
               COALESCE(w.label,'') AS workspace
        FROM session s JOIN source src ON src.id = s.source_id
        LEFT JOIN workspace w ON w.id = s.workspace_id
        WHERE s.id = ?""", (session_id,)).fetchone()
    if meta is None:
        typer.echo(f"no session #{session_id}")
        raise typer.Exit(1)

    when = (datetime.fromtimestamp(meta["started_at"] / 1000, timezone.utc)
            .strftime("%Y-%m-%d %H:%M") if meta["started_at"] else "?")
    typer.echo(f"{meta['title'] or '(untitled)'}")
    typer.echo(f"{when} · {meta['source']} · {meta['workspace'] or '-'} "
               f"· {meta['host'] or '-'}")
    typer.echo(f"{meta['raw_path']}\n" + "-" * 72)

    kinds = ("text", "thinking") if not tools else \
            ("text", "thinking", "tool_use", "tool_result")
    rows = con.execute(f"""
        SELECT m.role, m.seq, m.on_active_path, p.kind, p.text, p.tool_name
        FROM message m JOIN part p ON p.message_id = m.id
        WHERE m.session_id = ? AND p.kind IN ({','.join('?' * len(kinds))})
          {'' if abandoned else 'AND m.on_active_path = 1'}
        ORDER BY m.seq, p.seq""", (session_id, *kinds)).fetchall()

    for row in rows:
        mark = "" if row["on_active_path"] else " (abandoned)"
        label = row["kind"] if row["kind"] != "text" else row["role"]
        if row["kind"] == "tool_use":
            label = f"tool:{row['tool_name']}"
        body = (row["text"] or "").strip()
        if not body:
            continue
        typer.echo(f"\n[{label}{mark}]")
        typer.echo(body[:4000])


@app.command("export")
def export_cmd(
    session_id: int = typer.Argument(
        None, help="single session id; omit to batch-export by filter or --id"),
    fmt: str = typer.Option("html,md", "--format", "-f",
                            help="comma-separated: html, md"),
    out: Path = typer.Option(None, "--out", "-o",
                             help="directory for the export (zip path if --zip); "
                                  "default data/exports/"),
    no_redact: bool = typer.Option(False, "--no-redact",
                                   help="skip secret redaction (on by default, "
                                        "regardless of the archive's own redact setting)"),
    wide_redact: bool = typer.Option(False, "--wide-redact",
                                     help="also apply shape-based redaction rules"),
    tools: bool = typer.Option(True, "--tools/--no-tools"),
    abandoned: bool = typer.Option(True, "--abandoned/--no-abandoned"),
    workspace: str = typer.Option(None, "--workspace", "-w"),
    source: str = typer.Option(None, "--source", "-s"),
    participant: str = typer.Option(None, "--participant", "-p"),
    host: str = typer.Option(None, "--host"),
    tag: str = typer.Option(None, "--tag"),
    model_type: str = typer.Option(None, "--model-type"),
    ids: list[int] = typer.Option(None, "--id", help="repeatable; batch by explicit id"),
    zip_output: bool = typer.Option(False, "--zip", help="write a .zip instead of a directory"),
    data_dir: Path = typer.Option(None, "--data-dir"),
) -> None:
    """Export one session, or a filtered batch, as a self-contained HTML file and/or
    Markdown (plus an `assets/` folder for anything too large or too image-shaped to
    inline).

    Secrets are redacted by default regardless of the archive's own `redact` setting —
    this is content meant to leave the machine. Pass --no-redact for a trusted local copy.
    A single session with no filters/--id is a batch of one, using the same layout.
    """
    from .export import batch as export_batch

    formats = {f.strip().lower() for f in fmt.split(",") if f.strip()}
    bad = formats - {"html", "md"}
    if bad:
        typer.echo(f"unknown format(s): {', '.join(sorted(bad))} — use html and/or md")
        raise typer.Exit(1)
    if not formats:
        typer.echo("no format selected")
        raise typer.Exit(1)

    con, store, db_path = _open(data_dir)

    if session_id is not None:
        row = con.execute("SELECT id FROM session WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            typer.echo(f"no session #{session_id}")
            raise typer.Exit(1)
        session_ids = [session_id]
    else:
        session_ids = export_batch.select_session_ids(
            con, ids=ids or None, workspace=workspace, source=source,
            participant=participant, host=host, tag=tag, model_type=model_type)
        if not session_ids:
            typer.echo("no sessions matched — pass a session id, --id, or a filter "
                       "(--workspace/--source/--participant/--host/--tag/--model-type)")
            raise typer.Exit(1)

    if out is not None:
        dest = out
    elif zip_output:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        dest = db_path.parent / "exports" / f"export-{stamp}.zip"
    else:
        dest = db_path.parent / "exports"

    result = export_batch.export_batch(
        con, store.root, session_ids, dest, fmt=formats, redact=not no_redact,
        wide=wide_redact, tools=tools, abandoned=abandoned, zip_output=zip_output)

    if not result.sessions:
        typer.echo("nothing exported — no matching session had content to render")
        raise typer.Exit(1)
    if session_id is not None:
        where = dest if zip_output else dest / result.sessions[0]["folder"]
        typer.echo(f"exported session #{session_id} -> {where}")
    else:
        typer.echo(f"exported {result.count} session(s) -> {dest}")


@app.command()
def serve(
    port: int = typer.Option(8787, "--port", "-p"),
    data_dir: Path = typer.Option(None, "--data-dir"),
) -> None:
    """Open the archive in a browser at http://127.0.0.1:<port>."""
    import uvicorn

    from .web.app import create_app

    typer.echo(f"http://127.0.0.1:{port}\n")
    typer.echo("  loopback only — this database holds credentials that leaked into")
    typer.echo("  tool output. Do not bind it to 0.0.0.0 or put it behind a tunnel.\n")
    # host is hardcoded, not an option: making it configurable is how a private
    # archive ends up on a LAN.
    uvicorn.run(create_app(data_dir), host="127.0.0.1", port=port, log_level="warning")


@app.command("fetch-images")
def fetch_images_cmd(
    dry_run: bool = typer.Option(False, "--dry-run", help="count them, fetch nothing"),
    limit: int = typer.Option(None, "--limit", help="stop after this many"),
    data_dir: Path = typer.Option(None, "--data-dir"),
) -> None:
    """Download images that were archived as a URL only, into the blob store.

    T3 Chat records a generated image as a CDN link and no bytes, so the viewer has
    nothing to show. This is the one command in the archive that reaches the network on
    purpose — it is deliberately not part of `ingest` or `sync`, which run unattended.

    Safe to re-run: parts that already have bytes are skipped.
    """
    from .core import fetch_images

    con, blobs, _ = _open(data_dir)
    pending = len(fetch_images.candidates(con))
    if not pending:
        typer.echo("nothing to fetch — every image part already has its bytes.")
        return
    if dry_run:
        typer.echo(f"{pending} image part(s) would be fetched from their source URL.")
        return

    typer.echo(f"fetching {min(pending, limit or pending)} of {pending} image(s)…")
    result = fetch_images.run(con, blobs, limit=limit)
    typer.echo(f"  fetched {result.fetched} "
               f"({result.bytes_fetched / 1024 / 1024:.1f} MB)")
    if result.skipped:
        typer.echo(f"  skipped {result.skipped} (no URL in the reference)")
    for part_id, why in result.failed:
        typer.echo(f"  failed  part {part_id}: {why}")


@app.command()
def validate(
    session_id: int = typer.Argument(None, help="check only this session"),
    source: str = typer.Option(None, "--source", "-s", help="only this source kind"),
    verbose: bool = typer.Option(False, "--verbose", "-v",
                                 help="also print info-level findings"),
    data_dir: Path = typer.Option(None, "--data-dir"),
) -> None:
    """Check the persisted message DAG for structural corruption.

    Independent of ingest -- re-derives root/leaf/cycle shape from what `message`
    actually holds, so a DAG-resolution regression is caught even when the ingest run
    itself reported no errors (docs/phase0-findings.md §4b).
    """
    from .core import validate as dagcheck
    con, _, _ = _open(data_dir)
    if session_id is not None:
        head = con.execute("SELECT 1 FROM session WHERE id = ?", (session_id,)).fetchone()
        if head is None:
            typer.echo(f"no session #{session_id}")
            raise typer.Exit(1)
        findings = dagcheck.check_session(con, session_id)
    else:
        findings = dagcheck.check_all(con, source_kind=source)

    shown = findings if verbose else [f for f in findings if f.severity != "info"]
    for f in shown:
        typer.echo(f"{f.severity.upper():<8} session {f.session_id:<8} "
                   f"{f.source_kind:<12} {f.code:<24} {f.detail}")
    errors = sum(1 for f in findings if f.severity == "error")
    typer.echo(f"\n{len(findings)} finding(s), {errors} error(s)")
    if errors:
        raise typer.Exit(1)


@app.command()
def doctor(
    dag: bool = typer.Option(False, "--dag", help="also run the DAG structural validator"),
    data_dir: Path = typer.Option(None, "--data-dir"),
) -> None:
    """Report parse drift: record types no adapter handled, per run."""
    con, _, _ = _open(data_dir)
    rows = con.execute("""SELECT source_kind, finished_at, stats FROM ingest_run
                          ORDER BY id DESC LIMIT 10""").fetchall()
    if not rows:
        typer.echo("no ingest runs recorded yet")
    for r in rows:
        s = json.loads(r["stats"])
        typer.echo(f"\n{r['source_kind']}  ({s.get('seconds')}s)")
        typer.echo(f"  new {s.get('new')}  updated {s.get('updated')}  "
                   f"skipped {s.get('skipped')}  orphaned {s.get('orphaned')}")
        if s.get("errors"):
            typer.echo(f"  ERRORS  {s['errors']}")
        if s.get("unknown_types"):
            typer.echo("  unhandled types:")
            for k, v in list(s["unknown_types"].items())[:12]:
                typer.echo(f"    {k:<44} {v}")

    if dag:
        from .core import validate as dagcheck
        summary = dagcheck.summarize(dagcheck.check_all(con))
        typer.echo(f"\nDAG check -- {summary['total']} finding(s): "
                   f"{summary['by_severity']}")
        typer.echo("  run `llma validate` for detail")

# --------------------------------------------------------------- Phase 7

@app.command()
def sync(
    no_vectors: bool = typer.Option(False, "--no-vectors",
                                    help="keyword index only — seconds, not minutes"),
    log: Path = typer.Option(None, "--log",
                             help="append a timestamped record here (used by the "
                                  "scheduled task, which has no console)"),
    data_dir: Path = typer.Option(None, "--data-dir"),
) -> None:
    """Ingest every source, then rebuild the search index. Unattended-safe.

    One command rather than two because it has to be: ingesting renumbers part ids and
    both indexes are keyed on them, so an automated ingest without a re-index leaves
    keyword search matching the wrong rows with nothing on screen to say so.
    """
    from .core import sync as core_sync

    write = core_sync.logger(log)
    write("sync start")
    try:
        res = core_sync.run(data_dir, with_vectors=not no_vectors, log=write)
    except core_sync.Busy as exc:
        write(f"skipped — {exc}")
        typer.echo(str(exc))
        raise typer.Exit()          # a skipped run is a success, not a failure
    except Exception as exc:                                    # noqa: BLE001
        write(f"FAILED — {type(exc).__name__}: {exc}")
        raise

    line = (f"done in {res.seconds:.0f}s — {res.new} new, {res.updated} updated"
            + (f", errors {res.errors}" if res.errors else ""))
    write(line)
    typer.echo(line if log is None else f"{line}\n-> {log}")

    # Repeated on the console even when the detail went to a log file: the export you
    # have not taken is the one thing a sync cannot fix for you, so it should not be
    # the part that only exists in a file nobody opens.
    if res.stale_exports:
        typer.echo(f"\n{len(res.stale_exports)} export(s) need you:")
        for row in res.stale_exports:
            typer.echo(f"  {row['label']:16} {row['reason']}")
            typer.echo(f"  {'':16} {row['how']}")


@app.command()
def freshness(
    days: int = typer.Option(30, "--days", "-d",
                             help="how often a bulk export should be refreshed"),
    all_sources: bool = typer.Option(False, "--all",
                                     help="include live and per-chat sources"),
    data_dir: Path = typer.Option(None, "--data-dir"),
) -> None:
    """Which exports have gone stale — and which sources cannot go stale.

    Only bulk-export sources are ranked: one request refreshes the whole account, so
    "re-export every N days" is a real instruction there. Local stores are re-read on
    every ingest, and per-chat sources (§2.1) have no single action that makes them
    current, so both are listed under --all and never marked overdue.
    """
    from .core import freshness as fresh

    con, _, _ = _open(data_dir)
    rows = fresh.report(con, interval_days=days)
    summary = fresh.summary(rows)

    def age(row) -> str:
        value = row["export_age_days"]
        return "never" if value is None else f"{value:.0f}d"

    mark = {"missing": "!!", "overdue": "!!", "due": " ~",
            "current": " .", "manual": " -", "live": " -"}

    shown = [r for r in rows if all_sources or r["mode"] == fresh.BULK]
    typer.echo(f"{'':2} {'source':<16} {'sessions':>8} {'exported':>9}  state")
    for row in shown:
        typer.echo(f"{mark[row['state']]} {row['label']:<16} {row['sessions']:>8} "
                   f"{age(row):>9}  {row['state']} — {row['reason']}")
        if row.get("idle"):
            typer.echo(f"{'':29} {'':>9}  {row['idle']}")

    if summary["needs_action"]:
        typer.echo("\nDo these:")
        for row in summary["needs_action"]:
            typer.echo(f"  {row['label']:<16} {row['how']}")
    else:
        typer.echo("\nAll bulk exports are current.")

    # A note explains a row that wants attention. A retention warning is printed even
    # for a source that is perfectly current, because a rolling delete window does not
    # care how recently you exported.
    for row in shown:
        if row["note"] and row["state"] in ("missing", "overdue", "due", "manual"):
            typer.echo(f"\n  {row['label']}: {row['note']}")

    retention = [r for r in shown if r["retention"]]
    if retention:
        typer.echo("\nRetention — these delete themselves whatever you do:")
        for row in retention:
            typer.echo(f"  {row['label']:<16} {row['retention']}")


schedule_app = typer.Typer(help="Run `llma sync` on a schedule (Windows Task Scheduler).")
app.add_typer(schedule_app, name="schedule")


@schedule_app.command("install")
def schedule_install(
    at: str = typer.Option("21:00", "--at", help="daily start time, HH:MM 24-hour"),
    no_vectors: bool = typer.Option(False, "--no-vectors",
                                    help="nightly keyword-only build; embed by hand"),
    dry_run: bool = typer.Option(False, "--dry-run",
                                 help="print the task definition, register nothing"),
    data_dir: Path = typer.Option(None, "--data-dir"),
) -> None:
    """Register a daily task that re-ingests every source and rebuilds the index."""
    from .core import schedule as sched

    try:
        plan = sched.make_plan(data_dir, at=at, with_vectors=not no_vectors)
        xml = sched.build_xml(plan)          # validates --at before touching anything
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1)

    typer.echo(f"task       {sched.TASK_NAME}")
    typer.echo(f"runs       daily at {plan.at}, and on the next wake if the machine "
               f"was asleep")
    typer.echo(f"command    {plan.describe()}")
    typer.echo(f"log        {plan.log_path}")

    if dry_run:
        typer.echo("\n" + xml)
        raise typer.Exit()

    try:
        sched.install(plan)
    except sched.NotWindows as exc:
        typer.echo(f"\n{exc}")
        raise typer.Exit(1)
    except sched.SchedulerError as exc:
        typer.echo(f"\nTask Scheduler refused: {exc}")
        raise typer.Exit(1)
    typer.echo("\nregistered. Check it with: llma schedule status")


@schedule_app.command("status")
def schedule_status() -> None:
    """Is the task registered, when did it last run, and did that run succeed?"""
    from .core import schedule as sched

    try:
        info = sched.status()
    except sched.NotWindows as exc:
        typer.echo(str(exc))
        raise typer.Exit(1)
    except sched.SchedulerError as exc:
        typer.echo(f"Task Scheduler refused: {exc}")
        raise typer.Exit(1)

    if info is None:
        typer.echo("not registered — run: llma schedule install")
        raise typer.Exit()

    typer.echo(f"task        {sched.TASK_NAME}")
    typer.echo(f"state       {info.get('state')}")
    typer.echo(f"last run    {info.get('last_run') or 'never'}")
    typer.echo(f"next run    {info.get('next_run') or '-'}")

    code = info.get("last_result")
    # 267011 (0x41303) is "task has not yet run" — printing that as a failure on a
    # freshly installed task is how a health check gets ignored.
    if code == 0:
        typer.echo("last result 0 (ok)")
    elif code in (None, 267011):
        typer.echo("last result — (has not run yet)")
    else:
        typer.echo(f"last result {code} — FAILED; the sync log has the reason")
    typer.echo(f"command     {info.get('command')} {info.get('arguments')}")
    if info.get("missed_runs"):
        typer.echo(f"missed      {info['missed_runs']} run(s)")


@schedule_app.command("remove")
def schedule_remove() -> None:
    """Unregister the task. The archive and its indexes are untouched."""
    from .core import schedule as sched

    try:
        removed = sched.remove()
    except sched.NotWindows as exc:
        typer.echo(str(exc))
        raise typer.Exit(1)
    except sched.SchedulerError as exc:
        typer.echo(f"Task Scheduler refused: {exc}")
        raise typer.Exit(1)
    typer.echo("removed" if removed else "nothing registered")


@schedule_app.command("run")
def schedule_run() -> None:
    """Start the registered task now, the way the scheduler would."""
    from .core import schedule as sched

    try:
        sched.run_now()
    except (sched.NotWindows, sched.SchedulerError) as exc:
        typer.echo(str(exc))
        raise typer.Exit(1)
    typer.echo("started — output goes to the task's log, not here")


@app.command()
def redact(
    apply_now: bool = typer.Option(False, "--apply",
                                   help="rewrite part.text (irreversible in the DB)"),
    wide: bool = typer.Option(False, "--wide",
                              help="add shape-based rules (password=, bearer headers, "
                                   "URL credentials) — more coverage, some misfires"),
    enable: bool = typer.Option(False, "--enable",
                                help="run this on every ingest from now on"),
    disable: bool = typer.Option(False, "--disable"),
    blobs: bool = typer.Option(False, "--blobs",
                               help="also scan data/blobs/ — reported, never rewritten"),
    status: bool = typer.Option(False, "--status", help="what has been redacted so far"),
    data_dir: Path = typer.Option(None, "--data-dir"),
) -> None:
    """Find and replace secrets in stored message text. Dry run unless --apply.

    Replacement is `[redacted:<rule>:<8 hex of the secret's sha256>]`, so two different
    leaked keys stay distinguishable and you can still tell months later whether the key
    that leaked is the one you rotated — without the plaintext being kept anywhere.

    This rewrites the database only. The originals under `raw_path` still hold the
    secret and ingest re-reads them, which is why --enable exists: it stores the setting
    so every future ingest redacts before anything is indexed or embedded.
    """
    from .core import redact as core_redact

    con, blob_store, _ = _open(data_dir)

    if disable:
        core_redact.set_enabled(con, False)
        typer.echo("redaction disabled for future ingests. Text already redacted is "
                   "not restored — the plaintext only exists in the source files.")
        raise typer.Exit()
    if enable:
        core_redact.set_enabled(con, True, wide=wide)
        typer.echo(f"redaction enabled for every future ingest "
                   f"({'wide' if wide else 'default'} rules)")

    if status:
        info = core_redact.summary(con)
        typer.echo(f"enabled     {'yes' if info['enabled'] else 'no'}"
                   f"{' (wide rules)' if info.get('wide') else ''}")
        typer.echo(f"redactions  {info['rows']} in {info['sessions']} session(s), "
                   f"{info['distinct']} distinct secret(s)")
        for row in info["by_rule"]:
            typer.echo(f"  {row['rule']:<20} {row['n']:>5} ({row['k']} distinct)")
        raise typer.Exit()

    rules = core_redact.active_rules(wide)
    typer.echo(f"{len(rules)} rule(s), {'wide' if wide else 'default'} set")

    res = core_redact.apply(con, wide=wide, dry_run=not apply_now)
    verb = "redacted" if apply_now else "would redact"
    typer.echo(f"{verb} {res.secrets} secret(s) across {res.parts_changed} part(s) "
               f"of {res.parts_scanned} scanned  ({res.seconds:.1f}s)")
    for name, count in sorted(res.by_rule.items(), key=lambda kv: -kv[1]):
        typer.echo(f"  {name:<20} {count:>5}")

    if blobs:
        findings = core_redact.scan_blobs(blob_store.root, wide=wide)
        total = sum(f["hits"] for f in findings)
        typer.echo(f"\nblob store: {total} match(es) in {len(findings)} file(s) "
                   f"— reported only, never rewritten.")
        typer.echo("  Blobs are content-addressed by sha256; editing one invalidates "
                   "the hash every referring row was deduplicated on.")
        for finding in findings[:10]:
            typer.echo(f"  {finding['hits']:>4}  {finding['path']}")

    if not apply_now and res.secrets:
        typer.echo("\nNothing was changed. Re-run with --apply to rewrite, and "
                   "--enable to keep it applied on every ingest.")
    if apply_now and res.parts_changed:
        typer.echo("\nRun `llma index` — the keyword index still holds the "
                   "pre-redaction text.")


if __name__ == "__main__":
    app()
