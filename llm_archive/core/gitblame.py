"""The git half of `llma blame`: a line of code to the commit that wrote it.

`who-touched` goes from a path to the sessions that changed it, which answers "why is
this module shaped like this". The question a reader actually has, cursor on line 137,
is narrower: *these lines*, why. git blame answers half of it — line → commit → a
timestamp and a subject — and stops at a hash, because git never knew there was a
conversation. The archive holds the other half: which session was editing that file in
the window that ended at the commit, and which session ran the `git commit` itself.

This module only talks to git. It runs `blame --porcelain` for the range, reads each
commit's two timestamps, and works out the lower edge of every commit's window: the
previous commit that touched the same path. `api.blame_payload` does the join against
`touched_file` and `command`; nothing here opens the database.

**The window is (previous commit touching the file, this commit].** Not the parent
commit: an agent edits `foo.py` at 10:00, the user commits `bar.py` alone at 10:10, and
`foo.py` is finally committed at 11:30. The parent of that commit is the 10:10 one, and
a window starting there loses the 10:00 edit that is in the diff. The last commit that
changed *this file* is the edge that is right.

Both timestamps of a commit are used, because they move apart. A rebase rewrites the
committer time of every commit it replays while leaving the author time alone, so an
edit made before an amend or a rebase sits between the two. The window closes at the
later one and, for the previous commit, opens at the earlier one.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

UNCOMMITTED = "0" * 40

_HEADER = re.compile(r"^([0-9a-f]{40}) (\d+) (\d+)(?: (\d+))?$")
_SEP = "\x1f"
_LOG_FORMAT = _SEP.join(("%H", "%an", "%at", "%ct", "%s"))


class GitError(RuntimeError):
    """git is missing, the path is not in a repository, or blame refused it.

    One class, because every caller does the same thing with all three: print the
    message and stop. The message is git's own where it has one, since "no such path
    'x' in HEAD" is clearer than anything this module could paraphrase it into.
    """


@dataclass(frozen=True)
class Commit:
    sha: str
    author: str
    author_time: int        # epoch seconds
    committer_time: int
    summary: str
    # the path this commit knew the file by: differs from the current one after a
    # rename, and it is what a session editing it back then would have recorded
    filename: str

    @property
    def opened_at(self) -> int:
        return min(self.author_time, self.committer_time)

    @property
    def closed_at(self) -> int:
        return max(self.author_time, self.committer_time)

    @property
    def uncommitted(self) -> bool:
        return self.sha == UNCOMMITTED


@dataclass
class Blame:
    repo: Path                       # the checkout's top level, resolved
    rel: str                         # the file, relative to `repo`, forward slashes
    start: int
    end: int
    # final line number -> sha, in line order. Uncommitted lines carry UNCOMMITTED.
    lines: dict[int, str] = field(default_factory=dict)
    commits: dict[str, Commit] = field(default_factory=dict)
    # sha -> the previous commit that touched the file, or None for the one that
    # created it. UNCOMMITTED's previous is the newest commit of the file.
    previous: dict[str, Commit | None] = field(default_factory=dict)

    def ranges(self, sha: str) -> list[tuple[int, int]]:
        """The line runs `sha` owns inside the range, as inclusive (first, last)."""
        out: list[tuple[int, int]] = []
        for line, owner in self.lines.items():
            if owner != sha:
                continue
            if out and out[-1][1] == line - 1:
                out[-1] = (out[-1][0], line)
            else:
                out.append((line, line))
        return out

    def in_order(self) -> list[str]:
        """Shas by first appearance, so the report reads top to bottom like the file."""
        seen: list[str] = []
        for sha in self.lines.values():
            if sha not in seen:
                seen.append(sha)
        return seen


def _git(args: list[str], cwd: Path) -> str:
    # quotePath off, so a non-ASCII filename comes back as itself rather than as the
    # C-escaped octal that would otherwise need decoding on this side.
    try:
        proc = subprocess.run(
            ["git", "-c", "core.quotePath=false", *args], cwd=str(cwd),
            capture_output=True, check=False)
    except FileNotFoundError:
        raise GitError("git is not installed or not on PATH") from None
    except OSError as exc:
        raise GitError(f"could not run git: {exc}") from None
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise GitError(err.splitlines()[-1] if err else f"git {args[0]} failed")
    return proc.stdout.decode("utf-8", errors="replace")


def repo_root(path: Path) -> Path:
    """The top level of the checkout `path` is in.

    Asked from the file's directory, not the process's — the archive is queried from
    wherever the CLI or the MCP client happens to be, which is rarely the repo.
    """
    where = path if path.is_dir() else path.parent
    if not where.exists():
        raise GitError(f"no such directory: {where}")
    try:
        top = _git(["rev-parse", "--show-toplevel"], where).strip()
    except GitError as exc:
        if "not a git repository" in str(exc).lower():
            raise GitError(f"{path} is not inside a git repository") from None
        raise
    return Path(top).resolve()


def _parse_porcelain(text: str, repo: Path, rel: str,
                     start: int, end: int) -> Blame:
    """`git blame --porcelain` into a Blame.

    Every line of output is one of three things: a header (`<sha> <orig> <final>
    [<n>]`) that starts a blamed line, a tab-prefixed line of the file's content, or a
    `key value` attribute of the commit named by the most recent header. Attributes are
    only printed the first time a commit appears, so they are accumulated per sha.
    """
    blame = Blame(repo=repo, rel=rel, start=start, end=end)
    attrs: dict[str, dict[str, str]] = {}
    current: str | None = None

    for raw in text.splitlines():
        if raw.startswith("\t"):
            continue
        header = _HEADER.match(raw)
        if header:
            current = header.group(1)
            blame.lines[int(header.group(3))] = current
            attrs.setdefault(current, {})
            continue
        if current is None:
            continue
        key, _, value = raw.partition(" ")
        attrs[current][key] = value

    for sha, a in attrs.items():
        blame.commits[sha] = Commit(
            sha=sha,
            author=a.get("author", ""),
            author_time=int(a.get("author-time") or 0),
            committer_time=int(a.get("committer-time") or 0),
            summary=a.get("summary", ""),
            filename=a.get("filename") or rel,
        )
    return blame


def _parse_log(text: str, filename: str) -> list[Commit]:
    out = []
    for line in text.splitlines():
        fields = line.split(_SEP)
        if len(fields) != 5:
            continue
        sha, author, at, ct, summary = fields
        try:
            out.append(Commit(sha=sha, author=author, author_time=int(at),
                              committer_time=int(ct), summary=summary,
                              filename=filename))
        except ValueError:
            continue
    return out


def _file_history(repo: Path, filename: str, before: str | None = None) -> list[Commit]:
    """Every commit that changed `filename`, newest first.

    `before` narrows it to what is reachable from that commit's parent, for a commit
    that knew the file by another name and so is absent from the current path's log.
    """
    args = ["log", f"--format={_LOG_FORMAT}"]
    if before:
        args.append(f"{before}^")
    args += ["--", filename]
    try:
        return _parse_log(_git(args, repo), filename)
    except GitError:
        return []           # `<root>^` is not a revision; the file has no earlier history


def blame(path: Path, start: int | None = None, end: int | None = None) -> Blame:
    """Blame `path` (or lines start..end of it) and resolve each commit's window.

    Blames the working tree, as `git blame` does by default, so the line numbers are
    the ones in the editor — and a line edited since the last commit comes back owned
    by UNCOMMITTED, whose window opens at the file's newest commit and never closes.
    """
    path = Path(path)
    if not path.is_file():
        raise GitError(f"no such file: {path}")
    repo = repo_root(path)
    try:
        rel = path.resolve().relative_to(repo).as_posix()
    except ValueError:
        raise GitError(f"{path} is not inside {repo}") from None

    if start is not None and start < 1:
        raise GitError("line numbers start at 1")
    if start is not None and end is not None and end < start:
        raise GitError(f"line range {start},{end} is empty")

    args = ["blame", "--porcelain"]
    if start is not None:
        args += ["-L", f"{start},{end if end is not None else start}"]
    args += ["--", rel]
    out = _git(args, repo)

    lines = sorted(int(m.group(3)) for m in map(_HEADER.match, out.splitlines()) if m)
    if not lines:
        raise GitError(f"{rel} has no lines to blame")
    result = _parse_porcelain(out, repo, rel, start or lines[0], end or lines[-1])

    # The lower edge of every window in one `git log`, rather than one per commit.
    # Newest first, so the entry after a commit is the one that touched the file
    # before it. Time order rather than ancestry is the point: the window is a span
    # of wall-clock time in which an agent's edits could have landed in the diff.
    history = _file_history(repo, rel)
    position = {c.sha: i for i, c in enumerate(history)}
    for sha in result.commits:
        if sha == UNCOMMITTED:
            result.previous[sha] = history[0] if history else None
        elif sha in position:
            i = position[sha]
            result.previous[sha] = history[i + 1] if i + 1 < len(history) else None
        else:
            # Renamed since: the commit is in the old name's history, not this one's.
            older = _file_history(repo, result.commits[sha].filename, before=sha)
            result.previous[sha] = older[0] if older else None
    return result


def parse_range(spec: str | None) -> tuple[int | None, int | None]:
    """`-L` in git's own syntax: `10,20`, `10-20`, or a single `137`."""
    if not spec:
        return None, None
    text = spec.strip()
    m = re.fullmatch(r"(\d+)\s*[,\-]\s*(\d+)", text)
    if m:
        return int(m.group(1)), int(m.group(2))
    if text.isdigit():
        return int(text), int(text)
    raise ValueError(f"line range must be START,END or a line number, got {spec!r}")


def split_line_suffix(target: str) -> tuple[str, int | None, int | None]:
    """`api.py:137` and `api.py:120-140` into a path and a range.

    Only a suffix that is unambiguously a line spec is taken, so a Windows drive
    letter (`C:\\x`) and a path that simply has a colon in it are left alone.
    """
    m = re.fullmatch(r"(.+?):(\d+)(?:[-,](\d+))?", target)
    if not m:
        return target, None, None
    start = int(m.group(2))
    return m.group(1), start, int(m.group(3)) if m.group(3) else start
