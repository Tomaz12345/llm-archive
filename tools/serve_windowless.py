"""Start `llma serve` with no console window, for Windows Task Scheduler.

`pythonw.exe -m llm_archive.cli serve` looks like it should work — it is exactly what
the nightly sync task does — and it exits 1 before the port is ever bound. pythonw
gives the process no console, so `sys.stdout` and `sys.stderr` are **None**, and
uvicorn's logging config asks `sys.stderr.isatty()` while deciding whether to colourise.
`None.isatty()` raises, `dictConfig` fails, and the whole thing dies with no console to
say so. `sync` gets away with pythonw because `sync --log` writes through its own
`core.sync.logger` and never builds a logging config; `serve` does.

So give the streams somewhere real to go before anything imports uvicorn. `data/serve.log`
gets the same treatment `core.sync.logger` gives `data/sync.log` — appended, truncated
from the front past 1 MB — which also means a scheduled start that fails (port already
taken is the likely one) leaves a traceback behind instead of an exit code and nothing.

    .venv\\Scripts\\pythonw.exe tools\\serve_windowless.py [--port 8787]

Arguments are passed straight through to `llma serve`, so the port default lives in the
CLI and not in two places.
"""

from __future__ import annotations

import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# The task runs this by path under pythonw, which puts `tools/` on sys.path and not the
# project root — so the package has to be found the same way whether or not it is
# pip-installed into the venv.
sys.path.insert(0, str(ROOT))

from llm_archive.core.sync import LOG_MAX_BYTES  # noqa: E402

LOG = ROOT / "data" / "serve.log"


def _open_log():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    try:
        if LOG.stat().st_size > LOG_MAX_BYTES:
            tail = LOG.read_bytes()[-LOG_MAX_BYTES // 2:]
            LOG.write_bytes(b"[...truncated...]\n" + tail)
    except OSError:
        pass
    # Line buffered: a server that is killed rather than shut down should still have
    # written the line that says what it was doing.
    return LOG.open("a", buffering=1, encoding="utf-8", errors="replace")


def main() -> int:
    handle = _open_log()
    sys.stdout = sys.stderr = handle
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    handle.write(f"\n{stamp}Z serve start — pid {os.getpid()}\n")

    from llm_archive.cli import app

    try:
        app(["serve", *sys.argv[1:]])
    except SystemExit as exc:            # click exits this way on --help and on error
        return int(exc.code or 0)
    except BaseException:                # noqa: BLE001 — the log is the only witness
        traceback.print_exc(file=handle)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
