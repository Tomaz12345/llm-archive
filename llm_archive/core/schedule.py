"""Registering `llma sync` with Windows Task Scheduler.

The archive is only as current as its last ingest, and the four local sources
(§7 "the three local CLI sources", now four with the VS Code panel) change every time
you work. Running ingest by hand is exactly the chore that stops happening.

**Why PowerShell's ScheduledTasks module and not `schtasks.exe`.** `schtasks /Query`
prints *localised* field names and `/TR` needs a nested-quoting incantation that breaks
the moment a path contains a space — and every path here does (`C:\\Users\\...\\Documents
\\Projekti\\LLM_sessions_grouping`). `Register-ScheduledTask` takes the task as XML and
`Get-ScheduledTaskInfo` returns objects whose property names are the same in every
Windows display language, so neither the install nor the status read can be broken by a
Slovene locale.

**Three settings that decide whether this actually runs on a laptop:**

* `StartWhenAvailable` — the machine is asleep at 21:00 most nights. Without this the
  run is simply skipped and the archive silently stops updating; with it, the task fires
  on the next wake.
* `MultipleInstancesPolicy = IgnoreNew` — a full embed pass takes minutes. Two of them
  writing chunks at once is the one way this job can corrupt its own index.
* `DisallowStartIfOnBatteries = false` — the default is *true*, which on a laptop that
  is rarely plugged in means the task exists, reports "Ready" forever, and never runs.

The action points at `pythonw.exe` rather than `python.exe`: the console window from a
nightly `python -m` steals focus. Output is not lost — `llma sync --log` writes it to a
file, which is also the only thing left to read when a scheduled run fails.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

TASK_NAME = "LLM Archive Sync"
DEFAULT_AT = "21:00"

# A full CPU embed pass over this corpus is minutes, not hours; two hours is a
# runaway-guard, not a budget.
TIME_LIMIT = "PT2H"

# The second job: a weekly digest written to a file, so Monday morning starts with
# last week already on disk. It reads the archive the nightly sync keeps current and
# is scheduled after it, not with it -- a digest built from a sync still running
# would describe the week up to yesterday.
DIGEST_TASK_NAME = "LLM Archive Digest"
DIGEST_AT = "08:00"
DIGEST_ON = "Monday"
DIGEST_TIME_LIMIT = "PT10M"

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
            "Sunday")


class NotWindows(RuntimeError):
    """Raised on platforms with no Task Scheduler, with the local equivalent."""


class SchedulerError(RuntimeError):
    """PowerShell refused. Carries whatever it said on stderr."""


@dataclass(frozen=True)
class Plan:
    """Everything the task will do, resolved to absolute paths before anything runs.

    `job` is `sync` (daily) or `digest` (weekly, on `weekly_on`); the two differ in
    the command, the trigger and the time limit, and nothing else.
    """
    python: Path
    project_root: Path
    data_dir: Path | None
    log_path: Path
    at: str
    with_vectors: bool
    job: str = "sync"
    weekly_on: str | None = None       # digest only
    out_dir: Path | None = None        # digest only: where the .md files land

    @property
    def task_name(self) -> str:
        return DIGEST_TASK_NAME if self.job == "digest" else TASK_NAME

    @property
    def time_limit(self) -> str:
        return DIGEST_TIME_LIMIT if self.job == "digest" else TIME_LIMIT

    @property
    def arguments(self) -> str:
        if self.job == "digest":
            args = ["-m", "llm_archive.cli", "digest", "--since", "last-week",
                    "--format", "md", "--out", f'"{self.out_dir}"']
        else:
            args = ["-m", "llm_archive.cli", "sync", "--log", f'"{self.log_path}"']
        if self.data_dir is not None:
            args += ["--data-dir", f'"{self.data_dir}"']
        if self.job == "sync" and not self.with_vectors:
            args.append("--no-vectors")
        return " ".join(args)

    def describe(self) -> str:
        return f'"{self.python}" {self.arguments}'

    def when(self) -> str:
        if self.job == "digest":
            return f"weekly on {self.weekly_on} at {self.at}"
        return f"daily at {self.at}"


def _windowless_python() -> Path:
    """pythonw.exe beside this interpreter, or this interpreter if there is none."""
    exe = Path(sys.executable)
    quiet = exe.with_name("pythonw.exe")
    return quiet if quiet.exists() else exe


def make_plan(data_dir: Path | None = None, at: str = DEFAULT_AT,
              with_vectors: bool = True, log_path: Path | None = None) -> Plan:
    from . import ingest

    db_path, _ = ingest.default_paths(data_dir)
    root = Path(__file__).resolve().parent.parent.parent
    return Plan(
        python=_windowless_python(),
        project_root=root,
        # Resolved now and written into the task: a task holding a relative path runs
        # against whatever directory the scheduler happens to start it in.
        data_dir=data_dir.resolve() if data_dir is not None else None,
        log_path=(log_path or (db_path.parent / "sync.log")).resolve(),
        at=at,
        with_vectors=with_vectors,
    )


def make_digest_plan(data_dir: Path | None = None, at: str = DIGEST_AT,
                     on: str = DIGEST_ON, out_dir: Path | None = None) -> Plan:
    """The weekly digest task. `out_dir` defaults to `<data>/digests/`; the digest
    command names each file after the day it was written."""
    from . import ingest

    day = _validate_weekday(on)
    db_path, _ = ingest.default_paths(data_dir)
    base = make_plan(data_dir, at=at)
    return Plan(
        python=base.python, project_root=base.project_root, data_dir=base.data_dir,
        log_path=base.log_path, at=at, with_vectors=True, job="digest",
        weekly_on=day,
        out_dir=(out_dir or (db_path.parent / "digests")).resolve(),
    )


def _validate_weekday(on: str) -> str:
    for day in WEEKDAYS:
        if day.lower() == (on or "").strip().lower():
            return day
    raise ValueError(f"--on wants a weekday name, got {on!r}")


def _validate_time(at: str) -> str:
    try:
        hh, mm = at.split(":")
        hour, minute = int(hh), int(mm)
        if not (0 <= hour < 24 and 0 <= minute < 60):
            raise ValueError
    except (ValueError, AttributeError):
        raise ValueError(f"--at wants HH:MM in 24-hour time, got {at!r}") from None
    return f"{hour:02d}:{minute:02d}"


def build_xml(plan: Plan) -> str:
    """The Task Scheduler definition.

    No `<?xml ... encoding=...?>` declaration on purpose: `Register-ScheduledTask -Xml`
    takes a .NET string, and a declaration claiming UTF-16 on a string PowerShell has
    already decoded is rejected as a mismatch.
    """
    at = _validate_time(plan.at)
    # StartBoundary needs a date, and any past date works for a daily trigger — the
    # scheduler advances it to the next occurrence. A fixed one keeps the XML stable so
    # re-installing does not look like a change.
    start = f"2026-01-01T{at}:00"
    if plan.job == "digest":
        description = ("Write last week's digest of LLM sessions to a file.\n"
                       "Created by: llma schedule install --digest")
        schedule = (f"<ScheduleByWeek><DaysOfWeek><{plan.weekly_on}/></DaysOfWeek>"
                    f"<WeeksInterval>1</WeeksInterval></ScheduleByWeek>")
    else:
        description = ("Re-ingest local LLM session stores and rebuild the search "
                       "index.\nCreated by: llma schedule install")
        schedule = "<ScheduleByDay><DaysInterval>1</DaysInterval></ScheduleByDay>"
    return f"""<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>{description}</Description>
  </RegistrationInfo>
  <Triggers>
    <CalendarTrigger>
      <StartBoundary>{start}</StartBoundary>
      <Enabled>true</Enabled>
      {schedule}
    </CalendarTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <AllowHardTerminate>true</AllowHardTerminate>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <ExecutionTimeLimit>{plan.time_limit}</ExecutionTimeLimit>
    <Priority>7</Priority>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{plan.python}</Command>
      <Arguments>{plan.arguments}</Arguments>
      <WorkingDirectory>{plan.project_root}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"""


def _powershell(script: str) -> str:
    if os.name != "nt":
        raise NotWindows(
            "Task Scheduler is Windows-only. The equivalent elsewhere is a crontab "
            "line:\n    0 21 * * *  " + f"{sys.executable} -m llm_archive.cli sync")
    proc = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise SchedulerError((proc.stderr or proc.stdout or "").strip()
                            or f"powershell exited {proc.returncode}")
    return proc.stdout


def install(plan: Plan, task_name: str | None = None) -> None:
    """Register (or replace) the task. Replacing is the normal case — re-running
    `install` after moving the project is how the recorded paths get corrected."""
    task_name = task_name or plan.task_name
    xml = build_xml(plan)
    if plan.out_dir is not None:
        plan.out_dir.mkdir(parents=True, exist_ok=True)
    plan.log_path.parent.mkdir(parents=True, exist_ok=True)

    # Via a UTF-8 temp file rather than inline: the XML contains quotes and newlines,
    # and embedding it in a -Command string is the quoting bug this module exists to
    # avoid. NamedTemporaryFile(delete=False) because Windows will not let PowerShell
    # open a file this process still holds.
    handle = tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False,
                                         encoding="utf-8")
    try:
        handle.write(xml)
        handle.close()
        _powershell(
            f"$xml = Get-Content -Raw -Encoding UTF8 '{handle.name}'; "
            f"Register-ScheduledTask -TaskName '{task_name}' -Xml $xml -Force "
            f"| Out-Null")
    finally:
        try:
            os.unlink(handle.name)
        except OSError:
            pass


def status(task_name: str = TASK_NAME) -> dict | None:
    """What the scheduler knows about the task, or None if it is not registered.

    `LastTaskResult` is the value worth reading: 0 is success, and anything else is a
    run that failed with nothing on screen to say so.
    """
    out = _powershell(f"""
        $t = Get-ScheduledTask -TaskName '{task_name}' -ErrorAction SilentlyContinue
        if (-not $t) {{ 'null' }} else {{
          $i = $t | Get-ScheduledTaskInfo
          $a = $t.Actions | Select-Object -First 1
          [pscustomobject]@{{
            state        = [string]$t.State
            last_run     = if ($i.LastRunTime) {{ $i.LastRunTime.ToString('s') }} else {{ $null }}
            next_run     = if ($i.NextRunTime) {{ $i.NextRunTime.ToString('s') }} else {{ $null }}
            last_result  = $i.LastTaskResult
            missed_runs  = $i.NumberOfMissedRuns
            command      = $a.Execute
            arguments    = $a.Arguments
            working_dir  = $a.WorkingDirectory
          }} | ConvertTo-Json -Compress
        }}""")
    text = out.strip()
    if not text or text == "null":
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


def remove(task_name: str = TASK_NAME) -> bool:
    """Unregister. Returns False if there was nothing to remove."""
    if status(task_name) is None:
        return False
    _powershell(f"Unregister-ScheduledTask -TaskName '{task_name}' -Confirm:$false")
    return True


def run_now(task_name: str = TASK_NAME) -> None:
    _powershell(f"Start-ScheduledTask -TaskName '{task_name}'")
