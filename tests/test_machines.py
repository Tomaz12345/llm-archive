"""Bundles from machines the archive cannot reach.

What a failure here would mean, in order of how much it would cost: a credential packed
into a zip that then travels on a USB stick; a session filed under the wrong machine, or
the same machine filed twice; an old bundle rolling a transcript back; a transcript the
other machine's 30-day cleanup deleted disappearing from here as well.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import pytest

from llm_archive.core import db, ingest, intake, machines

T1 = time.mktime((2026, 9, 1, 12, 0, 0, 0, 0, -1))
T2 = time.mktime((2026, 9, 20, 12, 0, 0, 0, 0, -1))


# -- helpers ---------------------------------------------------------------


def record(uid, parent, role, text, ts):
    return {"type": role, "uuid": uid, "parentUuid": parent, "timestamp": ts,
            "cwd": "C:\\Users\\x\\Projekti\\demo",
            "message": {"role": role, "content": [{"type": "text", "text": text}]}}


def transcript(turns: int) -> str:
    out, parent = [], None
    for i in range(turns):
        uid = f"m{i}"
        out.append(record(uid, parent, "user" if i % 2 == 0 else "assistant",
                          f"turn {i}", f"2026-09-01T10:00:{i:02d}Z"))
        parent = uid
    return "\n".join(json.dumps(r) for r in out) + "\n"


def write(path: Path, body: str, mtime: float = T1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    os.utime(path, (mtime, mtime))
    return path


def fake_machine(base: Path) -> tuple[Path, Path]:
    """A home and an APPDATA holding every store, plus everything that must stay out."""
    home, appdata = base / "home", base / "appdata"
    claude = home / ".claude"
    project = claude / "projects" / "c--Users-x-Projekti-demo"
    write(project / "s1.jsonl", transcript(2))
    write(project / "s1" / "subagents" / "agent-a.jsonl", transcript(2))
    write(project / "s1" / "tool-results" / "abc123.txt", "FULL OUTPUT")
    write(project / "memory" / "notes.md", "a memory no adapter reads")
    write(claude / ".credentials.json", '{"claudeAiOauth": {"accessToken": "sk-ant-X"}}')
    write(claude / "settings.json", '{"cleanupPeriodDays": 90, "apiKeyHelper": "secret"}')
    write(claude / "history.jsonl", '{"display": "typed prompts"}\n')

    codex = home / ".codex"
    write(codex / "sessions" / "2026" / "09" / "01" / "rollout-1.jsonl", "{}\n")
    write(codex / "session_index.jsonl", '{"id": "1", "thread_name": "t"}\n')
    write(codex / "auth.json", '{"OPENAI_API_KEY": "sk-X"}')
    write(codex / "config.toml", "model = 'x'")

    opencode = home / ".local" / "share" / "opencode"
    write(opencode / "storage" / "session" / "p1" / "ses_1.json", '{"id": "ses_1"}')
    write(opencode / "storage" / "message" / "ses_1" / "msg_1.json", '{"id": "msg_1"}')
    write(opencode / "storage" / "part" / "msg_1" / "prt_1.json", '{"id": "prt_1"}')
    write(opencode / "storage" / "project" / "p1.json", '{"id": "p1"}')
    write(opencode / "auth.json", '{"anthropic": {"key": "sk-X"}}')

    user = appdata / "Code" / "User"
    write(user / "workspaceStorage" / "h1" / "workspace.json", '{"folder": "file:///c%3A/x"}')
    write(user / "workspaceStorage" / "h1" / "chatSessions" / "c1.json", '{"requests": []}')
    write(user / "workspaceStorage" / "h1" / "chatSessions" / "c2.jsonl", '{"kind": 0}\n')
    write(user / "workspaceStorage" / "h1" / "state.vscdb", "sqlite")
    write(user / "workspaceStorage" / "h2" / "workspace.json", '{"folder": "file:///c%3A/y"}')
    write(user / "globalStorage" / "emptyWindowChatSessions" / "e1.json", '{"requests": []}')
    write(user / "globalStorage" / "github.copilot" / "token.json", '{"token": "ghu_X"}')
    write(user / "settings.json", "{}")
    return home, appdata


def stores_of(home: Path, appdata: Path) -> dict[str, Path]:
    return {"claude_code": home / ".claude", "codex": home / ".codex",
            "opencode": home / ".local" / "share" / "opencode",
            "vscode_chat": appdata / "Code" / "User"}


EXPECTED = {
    "claude_code/projects/c--Users-x-Projekti-demo/s1.jsonl",
    "claude_code/projects/c--Users-x-Projekti-demo/s1/subagents/agent-a.jsonl",
    "claude_code/projects/c--Users-x-Projekti-demo/s1/tool-results/abc123.txt",
    "codex/sessions/2026/09/01/rollout-1.jsonl",
    "codex/session_index.jsonl",
    "opencode/storage/session/p1/ses_1.json",
    "opencode/storage/message/ses_1/msg_1.json",
    "opencode/storage/part/msg_1/prt_1.json",
    "vscode_chat/workspaceStorage/h1/workspace.json",
    "vscode_chat/workspaceStorage/h1/chatSessions/c1.json",
    "vscode_chat/workspaceStorage/h1/chatSessions/c2.jsonl",
    "vscode_chat/globalStorage/emptyWindowChatSessions/e1.json",
}


def claude_only(base: Path, body: str, mtime: float, name: str = "s1") -> dict:
    """Stores holding one Claude Code transcript, for the merge tests."""
    claude = base / ".claude"
    write(claude / "projects" / "c--Users-x-Projekti-demo" / f"{name}.jsonl", body, mtime)
    return {"claude_code": claude}


@pytest.fixture
def archive(tmp_path):
    con = db.connect(tmp_path / "data" / "archive.db")
    yield con, tmp_path
    con.close()


def bundle_zip(path: Path, manifest: dict | None, entries: dict[str, str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        if manifest is not None:
            zf.writestr(machines.MANIFEST, json.dumps(manifest))
        for name, body in entries.items():
            zf.writestr(name, body)
    return path


def manifest(host="LAPTOP-7Q2", packed_at="2026-10-06T12:00:00Z"):
    return {"format": machines.FORMAT, "version": 1, "host": host,
            "packed_at": packed_at, "sources": {}}


def files_under(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


# -- packing ---------------------------------------------------------------


def test_a_bundle_carries_transcripts_and_never_credentials(tmp_path):
    home, appdata = fake_machine(tmp_path)
    packed = machines.pack(tmp_path / "out" / "b.zip", host="LAPTOP-7Q2",
                           stores=stores_of(home, appdata))

    with zipfile.ZipFile(packed.path) as zf:
        names = set(zf.namelist()) - {machines.MANIFEST}
        body = b"".join(zf.read(n) for n in zf.namelist())
    assert names == EXPECTED
    assert b"sk-ant-X" not in body and b"sk-X" not in body and b"ghu_X" not in body
    assert b"secret" not in body, "only cleanupPeriodDays may be read from settings"
    assert packed.cleanup_period_days == 90


def test_the_manifest_names_the_machine_and_its_retention(tmp_path):
    home, appdata = fake_machine(tmp_path)
    packed = machines.pack(tmp_path / "b.zip", host="LAPTOP-7Q2",
                           stores=stores_of(home, appdata))
    with zipfile.ZipFile(packed.path) as zf:
        meta = json.loads(zf.read(machines.MANIFEST))
    assert meta["format"] == machines.FORMAT and meta["host"] == "LAPTOP-7Q2"
    assert meta["sources"]["claude_code"]["files"] == 3
    assert meta["sources"]["claude_code"]["cleanup_period_days"] == 90


def test_packing_nothing_is_an_error_not_an_empty_zip(tmp_path):
    with pytest.raises(machines.BundleError):
        machines.pack(tmp_path / "b.zip", host="x", stores={"claude_code": tmp_path / "no"})
    assert not (tmp_path / "b.zip").exists()


# -- taking one in ---------------------------------------------------------


def test_a_bundle_round_trips_into_sessions_under_its_own_host(archive):
    con, tmp_path = archive
    home, appdata = fake_machine(tmp_path / "laptop")
    packed = machines.pack(tmp_path / "usb" / "b.zip", host="LAPTOP-7Q2",
                           stores=stores_of(home, appdata))
    root = tmp_path / "data" / "machines"

    assert intake.identify(packed.path) == machines.KIND
    [taken] = intake.take_all([packed.path], con, tmp_path / "data" / "drops",
                              machines_root=root)
    assert (taken.action, taken.kind, taken.host) == ("added", "machine", "LAPTOP-7Q2")
    assert files_under(taken.dest) == EXPECTED | {machines.MACHINE_FILE}

    adapters = ingest.machine_adapters(None, root, host="LAPTOP-7Q2")
    claude = [a for a in adapters if a.kind == "claude_code"]
    res = ingest.run(claude[0], con, archive_drops=False)
    assert res.new >= 1
    rows = con.execute("SELECT host, raw_path FROM session").fetchall()
    assert {r["host"] for r in rows} == {"LAPTOP-7Q2"}
    assert all(Path(r["raw_path"]).is_relative_to(root) for r in rows), \
        "raw_path must point into the archive, not at the USB stick"


def test_the_same_bundle_twice_is_held(archive):
    con, tmp_path = archive
    path = machines.pack(tmp_path / "b.zip", host="LAPTOP-7Q2",
                         stores=claude_only(tmp_path / "h", transcript(2), T1)).path
    drops, root = tmp_path / "data" / "drops", tmp_path / "data" / "machines"
    intake.take_all([path], con, drops, machines_root=root)
    [again] = intake.take_all([path], con, drops, machines_root=root)
    assert (again.action, again.detail) == ("held", "already merged")


def test_every_sync_reads_the_filed_machines(tmp_path):
    root = tmp_path / "machines"
    machines.unpack(machines.pack(tmp_path / "b.zip", host="LAPTOP-7Q2",
                                  stores=claude_only(tmp_path / "h", transcript(2), T1)).path,
                    root)
    adapters = ingest.build_adapters(None, "claude_code", machines=root)
    filed = [a for a in adapters if getattr(a, "machine", None)]
    assert [(a.host, ingest.adapter_label(a)) for a in filed] == \
        [("LAPTOP-7Q2", "Claude Code @ LAPTOP-7Q2")]
    assert any(not getattr(a, "machine", None) for a in adapters), "the live store too"


# -- merging ---------------------------------------------------------------


def test_an_older_bundle_cannot_roll_a_transcript_back(tmp_path):
    root = tmp_path / "machines"
    old = machines.pack(tmp_path / "old.zip", host="LAPTOP-7Q2",
                        stores=claude_only(tmp_path / "a", transcript(2), T1)).path
    new = machines.pack(tmp_path / "new.zip", host="LAPTOP-7Q2",
                        stores=claude_only(tmp_path / "b", transcript(6), T2)).path

    machines.unpack(new, root)
    late = machines.unpack(old, root)

    stored = root / "laptop-7q2" / "claude_code" / "projects" / \
        "c--Users-x-Projekti-demo" / "s1.jsonl"
    assert stored.read_text(encoding="utf-8") == transcript(6)
    assert late.kept == 1 and not late.updated


def test_a_grown_transcript_replaces_the_shorter_copy(tmp_path):
    root = tmp_path / "machines"
    machines.unpack(machines.pack(tmp_path / "a.zip", host="L",
                                  stores=claude_only(tmp_path / "a", transcript(2), T1)).path,
                    root)
    grown = machines.unpack(machines.pack(
        tmp_path / "b.zip", host="L",
        stores=claude_only(tmp_path / "b", transcript(6), T2)).path, root)
    assert grown.updated == {"claude_code": 1}


def test_a_transcript_the_machine_deleted_is_kept_here(tmp_path):
    """Claude Code's 30-day cleanup on the laptop must not reach into the archive."""
    root = tmp_path / "machines"
    first = claude_only(tmp_path / "a", transcript(2), T1, name="gone-later")
    machines.unpack(machines.pack(tmp_path / "a.zip", host="L", stores=first).path, root)
    machines.unpack(machines.pack(
        tmp_path / "b.zip", host="L",
        stores=claude_only(tmp_path / "b", transcript(2), T2, name="newer")).path, root)

    project = root / "l" / "claude_code" / "projects" / "c--Users-x-Projekti-demo"
    assert {p.name for p in project.glob("*.jsonl")} == {"gone-later.jsonl", "newer.jsonl"}


def test_an_unchanged_bundle_rewrites_nothing(tmp_path):
    root = tmp_path / "machines"
    stores = claude_only(tmp_path / "a", transcript(2), T1)
    machines.unpack(machines.pack(tmp_path / "a.zip", host="L", stores=stores).path, root)
    again = machines.unpack(machines.pack(tmp_path / "b.zip", host="L", stores=stores).path,
                            root)
    assert (again.unchanged, again.added, again.updated) == (1, {}, {})


# -- naming the machine ----------------------------------------------------


def test_a_bare_claude_folder_zip_needs_a_host_and_keeps_only_transcripts(archive):
    con, tmp_path = archive
    path = bundle_zip(tmp_path / "claude-backup.zip", None, {
        ".claude/projects/c--Users-x-Projekti-demo/s1.jsonl": transcript(2),
        ".claude/projects/c--Users-x-Projekti-demo/memory/notes.md": "memory",
        ".claude/.credentials.json": '{"accessToken": "sk-ant-X"}',
        ".claude/settings.json": "{}",
    })
    drops, root = tmp_path / "data" / "drops", tmp_path / "data" / "machines"

    [nameless] = intake.take_all([path], con, drops, machines_root=root)
    assert nameless.action == "failed" and "--host" in nameless.detail
    assert intake.known(con, nameless.sha) is None, "re-sniffed once a --host is given"

    [named] = intake.take_all([path], con, drops, machines_root=root, host="laptop")
    assert named.action == "added" and named.host == "laptop"
    assert files_under(root / "laptop") == {
        "machine.json", "claude_code/projects/c--Users-x-Projekti-demo/s1.jsonl"}


def test_a_hostname_seen_once_finds_its_machine_again(tmp_path):
    """Filed as "laptop" once; the next bundle from that box needs no --host."""
    root = tmp_path / "machines"
    entries = {"claude_code/projects/p/s1.jsonl": transcript(2)}
    machines.unpack(bundle_zip(tmp_path / "a.zip", manifest("DESKTOP-AB12C"), entries),
                    root, host="laptop")
    second = machines.unpack(
        bundle_zip(tmp_path / "b.zip", manifest("DESKTOP-AB12C", "2026-10-20T12:00:00Z"),
                   {"claude_code/projects/p/s2.jsonl": transcript(2)}), root)
    assert second.host == "laptop"
    assert [m.host for m in machines.listing(root)] == ["laptop"]


def test_a_casing_slip_does_not_make_a_second_machine(tmp_path):
    root = tmp_path / "machines"
    entries = {"claude_code/projects/p/s1.jsonl": transcript(2)}
    machines.unpack(bundle_zip(tmp_path / "a.zip", None, {
        ".claude/projects/p/s1.jsonl": transcript(2)}), root, host="Laptop")
    machines.unpack(bundle_zip(tmp_path / "b.zip", manifest("laptop"), entries), root)
    assert [m.host for m in machines.listing(root)] == ["Laptop"]


def test_this_machines_own_bundle_is_refused(tmp_path):
    """Its sessions are read live; a second copy under another host would flip-flop."""
    root = tmp_path / "machines"
    entries = {"claude_code/projects/p/s1.jsonl": transcript(2)}
    here = ingest.local_host()
    with pytest.raises(machines.BundleError):
        machines.unpack(bundle_zip(tmp_path / "a.zip", manifest(here), entries), root)
    with pytest.raises(machines.BundleError):
        machines.unpack(bundle_zip(tmp_path / "b.zip", manifest(here), entries), root,
                        host="laptop")
    assert not root.exists()


# -- hostile or odd zips ---------------------------------------------------


def test_a_bundle_cannot_write_outside_its_machine(tmp_path):
    root = tmp_path / "data" / "machines"
    result = machines.unpack(bundle_zip(tmp_path / "evil.zip", manifest(), {
        "claude_code/projects/p/s1.jsonl": transcript(2),
        "claude_code/projects/../../../evil.jsonl": "x",
        "claude_code/projects/p/../../../../evil2.jsonl": "x",
        "claude_code/C:/evil3.jsonl": "x",
        "settings/x.jsonl": "x",
        "claude_code/.credentials.json": "x",
    }), root)
    assert not list(tmp_path.rglob("evil*.jsonl"))
    assert result.ignored == 5
    assert files_under(root / "laptop-7q2") == {
        "machine.json", "claude_code/projects/p/s1.jsonl"}


def test_backslash_entry_names_are_read_as_paths(tmp_path):
    """Windows PowerShell 5.1's zip writers have stored names with backslashes."""
    root = tmp_path / "machines"
    machines.unpack(bundle_zip(tmp_path / "ps.zip", manifest(), {
        "claude_code\\projects\\p\\s1.jsonl": transcript(2)}), root)
    assert (root / "laptop-7q2" / "claude_code" / "projects" / "p" / "s1.jsonl").exists()


def test_a_newer_bundle_format_is_refused_by_name(tmp_path):
    meta = manifest() | {"version": machines.VERSION + 1}
    with pytest.raises(machines.BundleError, match="update llma"):
        machines.unpack(bundle_zip(tmp_path / "a.zip", meta, {
            "claude_code/projects/p/s1.jsonl": "x"}), tmp_path / "m")


def test_web_exports_are_still_identified_as_web_exports(tmp_path):
    path = tmp_path / "conversations-000.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("conversations.json", "[]")
        zf.writestr("projects.json", "[]")
    assert machines.inspect(path) is None


# -- due -------------------------------------------------------------------


def test_a_bundle_is_due_a_week_before_claude_would_delete(tmp_path):
    root = tmp_path / "machines"
    machines.unpack(bundle_zip(tmp_path / "a.zip", manifest(), {
        "claude_code/projects/p/s1.jsonl": transcript(2)}), root)
    packed = machines.listing(root)[0].packed_at

    assert machines.interval_days(machines.listing(root)[0]) == 23
    assert not machines.due(root, now_ms=packed + 22 * machines.DAY_MS)
    [row] = machines.due(root, now_ms=packed + 24 * machines.DAY_MS)
    assert row["host"] == "LAPTOP-7Q2" and "30d after its last activity" in row["reason"]


def test_a_long_retention_still_wants_a_bundle_monthly(tmp_path):
    root = tmp_path / "machines"
    meta = manifest() | {"sources": {"claude_code": {"cleanup_period_days": 365}}}
    machines.unpack(bundle_zip(tmp_path / "a.zip", meta, {
        "claude_code/projects/p/s1.jsonl": transcript(2)}), root)
    assert machines.interval_days(machines.listing(root)[0]) == 30


# -- the nightly sync ------------------------------------------------------


def test_the_nightly_sync_reads_a_filed_machine_and_names_a_late_one(tmp_path, monkeypatch):
    """The scheduled task is the only reader most bundles will ever get."""
    from llm_archive.core import sync

    data = tmp_path / "data"
    db.connect(data / "archive.db").close()
    machines.unpack(bundle_zip(tmp_path / "a.zip", manifest(packed_at="2020-01-01T00:00:00Z"),
                               {"claude_code/projects/p/s1.jsonl": transcript(2)}),
                    ingest.machines_dir(data))

    # Only the filed machine: the live stores here belong to whoever runs the tests.
    seen = {}

    def filed_only(blobs, *args, machines=None, **kwargs):
        assert machines is not None, "sync must hand build_adapters the machines folder"
        seen["machines"] = machines
        return ingest.machine_adapters(blobs, machines)

    monkeypatch.setattr(ingest, "build_adapters", filed_only)

    lines: list[str] = []
    res = sync.run(data, with_vectors=False, log=lines.append)

    assert seen["machines"] == ingest.machines_dir(data)
    assert res.new == 1
    con = db.connect(data / "archive.db")
    assert [r["host"] for r in con.execute("SELECT host FROM session")] == ["LAPTOP-7Q2"]
    con.close()
    assert "  claude_code @ LAPTOP-7Q2: +1 new, 0 updated, 0 unchanged" in lines
    assert [r["host"] for r in res.stale_machines] == ["LAPTOP-7Q2"]
    assert any(line.startswith("  BUNDLE DUE  LAPTOP-7Q2: last bundle packed")
               for line in lines)

    again = sync.run(data, with_vectors=False, log=lambda _: None)
    assert (again.new, again.updated) == (0, 0), "an unchanged tree is skipped, not re-read"


# -- the CLI and the PowerShell packer -------------------------------------


def test_llma_add_ingests_a_bundle_under_its_machine(tmp_path):
    from typer.testing import CliRunner
    from llm_archive.cli import app as cli_app

    home, appdata = fake_machine(tmp_path / "laptop")
    zip_path = machines.pack(tmp_path / "b.zip", host="LAPTOP-7Q2",
                             stores=stores_of(home, appdata)).path
    data = str(tmp_path / "data")

    r = CliRunner().invoke(cli_app, ["add", str(zip_path), "--data-dir", data])
    assert r.exit_code == 0, r.output
    assert "machine" in r.output and "=== Claude Code @ LAPTOP-7Q2 ===" in r.output

    r = CliRunner().invoke(cli_app, ["machines", "--data-dir", data])
    assert r.exit_code == 0, r.output
    assert "LAPTOP-7Q2" in r.output and "keeps a transcript 90d" in r.output


def test_pack_script_writes_the_script(tmp_path):
    from typer.testing import CliRunner
    from llm_archive.cli import app as cli_app

    r = CliRunner().invoke(cli_app, ["pack", "--script", str(tmp_path / "usb")])
    assert r.exit_code == 0, r.output
    script = (tmp_path / "usb" / "pack-machine.ps1").read_bytes()
    assert script == machines.script_path().read_bytes()
    assert max(script) < 128, "PowerShell 5.1 reads a BOM-less script as ANSI"


@pytest.mark.skipif(os.name != "nt" or shutil.which("powershell.exe") is None,
                    reason="Windows PowerShell is what the script is written for")
def test_the_powershell_packer_packs_what_llma_pack_packs(tmp_path):
    """The script re-states RULES; this is what keeps the two from drifting apart."""
    home, appdata = fake_machine(tmp_path)
    env = os.environ | {"USERPROFILE": str(home), "APPDATA": str(appdata)}
    out = tmp_path / "out"
    proc = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-File", str(machines.script_path()), "-OutDir", str(out),
         "-MachineName", "LAPTOP-7Q2"],
        env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    [zip_path] = out.glob("llma-machine-laptop-7q2-*.zip")
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        meta = json.loads(zf.read(machines.MANIFEST))
    assert names - {machines.MANIFEST} == EXPECTED
    assert meta["host"] == "LAPTOP-7Q2" and meta["version"] == machines.VERSION
    assert meta["sources"]["claude_code"]["cleanup_period_days"] == 90

    # ...and what it wrote is something this side takes in.
    result = machines.unpack(zip_path, tmp_path / "machines")
    assert sum(result.added.values()) == len(EXPECTED) and not result.ignored
