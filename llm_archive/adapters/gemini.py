"""Gemini adapter — reads the Google Takeout "My Activity" export for Gemini Apps.

Export flow: takeout.google.com -> deselect all -> **My Activity** -> *Multiple formats*
-> Activity records: **HTML** -> limit to *Gemini Apps* -> export. Arrives as
`takeout-<instant>-<n>-<part>.zip`:

    takeout-*.zip
      └── Takeout/<My Activity>/<Gemini Apps>/
            ├── <MyActivity>.html      every recorded activity, newest first
            └── slika-<hex>.png|jpg    the files you attached to a prompt

Every path component in there is **localised** — this account exports Slovene, so the
folder is `Moja dejavnost/Aplikacije Gemini` and the document is `Moja_dejavnost.html`.
Nothing here matches on those names: discovery sniffs the HTML for the MyActivity cell
markup plus a Gemini fingerprint, so a re-export in another language still lands.

Shape — a Material Design page, one `outer-cell` div per activity, 745 of them here:

    <div class="outer-cell">
      <div class="header-cell">      Aplikacije Gemini            <- the product
      <div class="content-cell ... body-1">
          Poslali ste poziv:<NBSP><the prompt>                     <- activity + payload
          1 priložena datoteka.  -  <a href="slika-x.png">…</a>    <- attachments (opt.)
          5. nov. 2025, 01:34:47 CEST                              <- the ONLY timestamp
          <p>…the answer, as rendered HTML…</p>                    <- the reply (opt.)
      <div class="content-cell ... text-right">  <img …>           <- attachment previews
      <div class="content-cell ... caption">
          Podrobnosti: https://gemini.google.com/app/<conv id>     <- the thread (opt.)

Seven things this format does that no other source in this archive does:

1. **It is an activity log, not a conversation store.** There are no conversation
   records at all — only stamped events, newest first, each carrying a link back to the
   thread it happened in. Sessions are *reconstructed* by grouping cells on that
   `gemini.google.com/app/<id>` link and sorting by time: 745 cells become 84 threads.

2. **The timestamp is the record separator.** A cell is `<activity> · <timestamp> ·
   <reply>` in one flat div, and the reply is not tagged in any way — the timestamp is
   the only boundary between what you asked and what Gemini answered. Exactly one
   matches in every one of the 745 cells, so the split is unambiguous; a cell where the
   pattern does not match is counted rather than guessed at, because a mis-split would
   file the model's answer under your name.

3. **Nothing labels the roles, the model, or the tokens.** No role field, no model
   name, no usage, no cost — anywhere. Role comes from position around the timestamp;
   `model_primary` stays NULL rather than being invented from the app name, and the
   token and cost columns are a real gap for this source, not a zero.

4. **The activity labels are prose in your export language.** "Poslali ste poziv:",
   "Ustvarjeno platno Gemini z naslovom", "Uporabljena je bila funkcija Pomočnika".
   Matching those would break on the next locale, so the parser matches *punctuation*
   instead: Google separates a label from its payload with a non-breaking space after
   the colon, and that `:\xa0` is what says "this cell is a prompt". Labels are recorded
   verbatim in `meta.activities`, so drift shows up in `llma doctor` instead of
   silently dropping content.

5. **Canvas documents have no thread link.** 52 cells are `Ustvarjeno platno …` — a
   generated document, sometimes a whole source file — and they carry no `Podrobnosti`
   URL at all. They are kept as one single-message session each, keyed by content hash,
   rather than discarded for the crime of not belonging to a thread.

6. **Attachments ship as real bytes.** Alone among the eight sources, this export
   contains the files you uploaded. They go into the blob store; the part records the
   filename and the true size.

7. **Every timestamp claims the export's own UTC offset.** All 745 stamps here read
   `CEST` (+02:00) — including the November and December ones, which were CET (+01:00)
   when they happened. Google formats the whole log in the offset that was current when
   you clicked export. The abbreviation is honoured as written, so re-exporting in
   winter would shift these instants by an hour; `meta.tz` keeps the labels that were
   seen so that stays visible rather than mysterious.

Retention matters here in a way it does not for a real export: My Activity auto-deletes
on a rolling window (18 months by default), so this file is a *window*, not an archive.
Message ids are derived from timestamps rather than from position, so a later export
that has lost its oldest cells still lines up with what is already stored (§8.7).
"""

from __future__ import annotations

import hashlib
import re
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterator

from ..core.blobs import BlobStore
from ..core.models import (
    INLINE_LIMIT,
    KIND_ATTACHMENT,
    KIND_IMAGE,
    KIND_TEXT,
    Message,
    ParseStats,
    Part,
    Session,
)
from ._drops import by_recency, candidates, taken_at

DROPS = Path(__file__).resolve().parent.parent.parent / "data" / "drops"

# The MyActivity stylesheet is ~140 KB of inlined Material Design, so a 64 KB sniff
# window would see nothing but CSS. 1 MB clears the preamble and the first few cells.
SNIFF_BYTES = 1 << 20

CELL_MARKER = '<div class="outer-cell'
GEMINI_URL = re.compile(r"https://gemini\.google\.com/(?:u/\d+/)?app/([0-9a-zA-Z_-]+)")
# Either the thread links, or the product named in the header cell — "Aplikacije
# Gemini", "Gemini Apps". A canvas-only log has no links at all, so the header has to
# be enough on its own; matching the word anywhere in the file would claim a Search
# log that happens to mention Gemini.
GEMINI_HINT = re.compile(
    r"gemini\.google\.com|mdl-typography--title\">[^<]{0,80}Gemini", re.I)

# The first `body-1` div is the activity; the second is `body-1 mdl-typography--
# text-right` (attachment previews) and does not match this.
BODY = re.compile(r'mdl-typography--body-1">(.*?)</div>', re.S)

# "5. nov. 2025, 01:34:47 CEST" — day first, month as a localised abbreviation.
TS_DAY_FIRST = re.compile(
    r"(?P<day>\d{1,2})\.\s*(?P<mon>[^\s.,<>]{3,12})\.?\s*(?P<year>\d{4}),\s*"
    r"(?P<h>\d{1,2}):(?P<m>\d{2}):(?P<s>\d{2})\s*(?P<tz>[A-Z]{2,5})")
# "Nov 5, 2025, 1:34:47 AM CEST" — what an English-locale export writes instead.
TS_MONTH_FIRST = re.compile(
    r"(?P<mon>[A-Za-z]{3,12})\.?\s+(?P<day>\d{1,2}),\s*(?P<year>\d{4}),\s*"
    r"(?P<h>\d{1,2}):(?P<m>\d{2}):(?P<s>\d{2})\s*(?P<ampm>[AP]M)?\s*(?P<tz>[A-Z]{2,5})")

# Slovene and English abbreviations, which differ in exactly three months.
MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "maj": 5, "may": 5, "jun": 6,
          "jul": 7, "avg": 8, "aug": 8, "sep": 9, "okt": 10, "oct": 10, "nov": 11,
          "dec": 12}

# Offsets for the abbreviations Google actually prints. Point 7: the label is trusted
# as written, because the alternative is guessing which zone the account was in.
ZONES = {"UTC": 0, "GMT": 0, "Z": 0, "WET": 0, "WEST": 1, "BST": 1, "CET": 1,
         "CEST": 2, "EET": 2, "EEST": 3, "MSK": 3,
         "EST": -5, "EDT": -4, "CST": -6, "CDT": -5, "MST": -7, "MDT": -6,
         "PST": -8, "PDT": -7, "AKST": -9, "AKDT": -8, "HST": -10,
         "IST": 5.5, "JST": 9, "AEST": 10, "AEDT": 11, "NZST": 12, "NZDT": 13}

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".heic", ".avif"}

# Google separates a label from its payload with NBSP — as the literal character in the
# activity line, as the entity in the caption block. Both forms are accepted.
LABEL_SEPARATORS = ("\xa0", "&nbsp;")


@dataclass(slots=True)
class _Cell:
    """One `outer-cell`, already split on its timestamp."""
    index: int                     # position in the document: newest is 0
    at: int                        # epoch ms, UTC
    tz: str
    label: str                     # the localised activity label, verbatim
    subject: str                   # what the label names: a canvas title, or ''
    conv_id: str | None
    asked: bool                    # did *you* speak in this cell, or only Gemini?
    prompt: str                    # markdown text, '' for a non-prompt activity
    reply: str
    attachments: list[str] = field(default_factory=list)
    raw: str = ""

    @property
    def is_prompt(self) -> bool:
        return self.asked and bool(self.prompt or self.attachments)


class _Reader:
    """A drop, read the same way whether it is a ZIP, a folder, or a bare HTML file.

    The attachments sit *beside* the document, so reading a member is never enough —
    the adapter also has to resolve `slika-x.png` relative to it.
    """

    def __init__(self, path: Path):
        self.path = path
        self._zip = zipfile.ZipFile(path) if path.suffix.lower() == ".zip" else None

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()

    def documents(self) -> list[str]:
        """Member names of every HTML file in the drop."""
        if self._zip is not None:
            return [n for n in self._zip.namelist() if n.lower().endswith(".html")]
        if self.path.is_dir():
            return sorted(str(p.relative_to(self.path)).replace("\\", "/")
                          for p in self.path.rglob("*.html"))
        return [self.path.name] if self.path.suffix.lower() == ".html" else []

    def read(self, member: str, limit: int | None = None) -> bytes | None:
        try:
            if self._zip is not None:
                with self._zip.open(member) as fh:
                    return fh.read(limit) if limit else fh.read()
            # A bare .html drop still has siblings: they live next to it, so members
            # resolve against the containing folder, not against the file itself.
            base = self.path.parent if self.path.is_file() else self.path
            target = base / member
            with target.open("rb") as fh:
                return fh.read(limit) if limit else fh.read()
        except (OSError, KeyError, zipfile.BadZipFile, RuntimeError, ValueError):
            return None

    def sibling(self, member: str, name: str) -> bytes | None:
        """A file next to `member` — an attachment referenced by a relative href."""
        folder = member.rsplit("/", 1)[0] if "/" in member else ""
        return self.read(f"{folder}/{name}" if folder else name)

    def text(self, member: str, limit: int | None = None) -> str:
        raw = self.read(member, limit)
        return "" if raw is None else raw.decode("utf-8", errors="replace")


class GeminiAdapter:
    kind = "gemini"
    label = "Gemini"
    surface = "web"

    def __init__(self, drops: Path | None = None, blobs: BlobStore | None = None):
        self.drops = drops or DROPS
        self.blobs = blobs

    # -- discovery ---------------------------------------------------------

    @staticmethod
    def claims(path: Path) -> bool:
        """Does this drop hold a Gemini Apps activity log? Markup, never the name.

        A Takeout ZIP can hold a dozen other products' activity logs; only the members
        that carry both the MyActivity cell markup and a Gemini fingerprint count, so
        exporting everything by accident costs a scan, not a wrong ingest.
        """
        if path.suffix.lower() not in (".zip", ".html") and not path.is_dir():
            return False
        reader = None
        try:
            reader = _Reader(path)
            return any(GeminiAdapter._is_gemini(reader.text(m, SNIFF_BYTES))
                       for m in reader.documents())
        except (OSError, zipfile.BadZipFile, RuntimeError):
            return False          # not a readable ZIP: a "not mine", not an error
        finally:
            if reader is not None:
                reader.close()

    def discover(self) -> list[Path]:
        return by_recency(p for p in candidates(self.drops) if self.claims(p))

    @staticmethod
    def _is_gemini(head: str) -> bool:
        return CELL_MARKER in head and bool(GEMINI_HINT.search(head))

    # -- parsing -----------------------------------------------------------

    def parse(self, path: Path, stats: ParseStats) -> Iterator[Session]:
        reader = _Reader(path)
        try:
            for member in reader.documents():
                if not self._is_gemini(reader.text(member, SNIFF_BYTES)):
                    continue
                stats.files += 1
                yield from self._parse_document(reader, member, stats)
        except (OSError, zipfile.BadZipFile, RuntimeError):
            stats.error(f"unreadable:{path.name}")
        finally:
            reader.close()

    def _parse_document(self, reader: _Reader, member: str,
                        stats: ParseStats) -> Iterator[Session]:
        raw = reader.read(member)
        if raw is None:
            stats.error(f"unreadable:{member.rsplit('/', 1)[-1]}")
            return
        digest = hashlib.sha256(raw).hexdigest()
        document = raw.decode("utf-8", errors="replace")

        # Grouped, not streamed: a thread's turns are scattered through the log, and
        # only the conversation link brings them back together.
        threads: dict[str, list[_Cell]] = {}
        for index, chunk in enumerate(self._split(document)):
            cell = self._cell(chunk, index, stats)
            if cell is None:
                continue
            key = cell.conv_id or self._loose_key(cell, stats)
            if key is None:
                continue
            threads.setdefault(key, []).append(cell)

        for key, cells in threads.items():
            # Document order is newest first, so an equal-second tie is resolved by
            # taking the *later* cell as the earlier turn.
            cells.sort(key=lambda c: (c.at, -c.index))
            session = self._session(reader, member, digest, key, cells, stats)
            if session is not None:
                stats.sessions += 1
                yield session

    @staticmethod
    def _split(document: str) -> list[str]:
        starts = [m.start() for m in re.finditer(re.escape(CELL_MARKER), document)]
        starts.append(len(document))
        return [document[starts[i]:starts[i + 1]] for i in range(len(starts) - 1)]

    def _loose_key(self, cell: _Cell, stats: ParseStats) -> str | None:
        """A key for a cell with no thread link — a canvas document, or nothing.

        Keyed by content so a re-export files the same canvas under the same session
        instead of creating a second one; the timestamp keeps two identical drafts
        apart.
        """
        if not cell.reply and not cell.prompt:
            stats.unknown(f"{self.kind}:activity-without-content")
            return None
        body = (cell.label + cell.prompt + cell.reply).encode("utf-8", "replace")
        return f"loose:{cell.at}:{hashlib.sha256(body).hexdigest()[:12]}"

    # -- one cell ----------------------------------------------------------

    def _cell(self, chunk: str, index: int, stats: ParseStats) -> _Cell | None:
        found = BODY.search(chunk)
        if found is None:
            stats.unknown(f"{self.kind}:cell-without-body")
            return None
        inner = found.group(1)

        stamp = self._stamp(inner, stats)
        if stamp is None:
            # Point 2: without the timestamp there is no boundary between the prompt
            # and the answer, and a guess would misattribute one to the other.
            stats.unknown(f"{self.kind}:cell-without-timestamp")
            return None
        at, tz, span = stamp

        before, after = inner[:span[0]], inner[span[1]:]
        label, subject, payload, attachments = self._activity(before)
        conv = GEMINI_URL.search(chunk)
        asked = label.endswith(":")

        # A prompt cell is you, then Gemini, split on the timestamp. An activity cell
        # is Gemini alone: whatever it produced sits *above* the timestamp (a canvas
        # document) and nothing of yours is in the cell at all.
        if asked:
            prompt, reply = to_text(f"{subject}<br>{payload}"), to_text(after)
        else:
            prompt = ""
            reply = "\n\n".join(x for x in (to_text(payload), to_text(after)) if x)

        return _Cell(index=index, at=at, tz=tz, label=label,
                     subject="" if asked else to_text(subject).strip(),
                     conv_id=conv.group(1) if conv else None, asked=asked,
                     prompt=prompt, reply=reply, attachments=attachments, raw=chunk)

    def _stamp(self, inner: str, stats: ParseStats) -> tuple[int, str, tuple] | None:
        """Find the one stamp that separates the prompt from the answer.

        Google gives the timestamp a `<br>`-delimited line of its own, so a line that
        is *nothing but* a date is the separator — which keeps a date pasted into a
        prompt from splitting the cell in the wrong place. A log that stops doing that
        falls back to the first parseable stamp anywhere in the cell.
        """
        offset = 0
        dated_line = False
        for segment in inner.split("<br>"):
            line = segment.strip()
            for pattern in (TS_DAY_FIRST, TS_MONTH_FIRST):
                match = pattern.fullmatch(line)
                if match is None:
                    continue
                dated_line = True
                moment = self._to_epoch(match, stats)
                if moment is not None:
                    start = offset + segment.index(line)
                    return moment[0], moment[1], (start, start + len(line))
            offset += len(segment) + len("<br>")

        if dated_line:
            # The separator is where it should be and cannot be read — whatever made it
            # unreadable is already counted, and scanning the prose would only find a
            # date that is not the separator.
            return None

        for pattern in (TS_DAY_FIRST, TS_MONTH_FIRST):
            for match in pattern.finditer(inner):
                moment = self._to_epoch(match, stats)
                if moment is not None:
                    stats.unknown(f"{self.kind}:timestamp-not-on-its-own-line")
                    return moment[0], moment[1], match.span()
        return None

    def _to_epoch(self, match: re.Match, stats: ParseStats) -> tuple[int, str] | None:
        month = MONTHS.get(match.group("mon").lower()[:3])
        if month is None:
            stats.unknown(f"{self.kind}:month:{match.group('mon')[:12]}")
            return None
        hour = int(match.group("h"))
        ampm = match.groupdict().get("ampm")
        if ampm:                                    # English exports print a 12h clock
            hour = (hour % 12) + (12 if ampm.upper() == "PM" else 0)
        tz = match.group("tz")
        if tz not in ZONES:
            stats.unknown(f"{self.kind}:timezone:{tz}")
        offset = ZONES.get(tz, 0)
        try:
            moment = datetime(int(match.group("year")), month, int(match.group("day")),
                              hour, int(match.group("m")), int(match.group("s")),
                              tzinfo=timezone(timedelta(hours=offset)))
        except ValueError:
            return None
        return int(moment.timestamp() * 1000), tz

    def _activity(self, before: str) -> tuple[str, str, str, list[str]]:
        """Split the pre-timestamp half into label, subject, body and attachments.

        Point 4: the label is not matched by its words. Google separates a label from
        what it names with a non-breaking space — "Poslali ste poziv:<NBSP>how do I…",
        "Ustvarjeno platno Gemini z naslovom<NBSP>Deep Learning Quiz" — and puts a
        colon there only when the payload is something *you* wrote. So the NBSP finds
        the boundary and the colon says who spoke, in any export language.
        """
        segments = before.split("<br>")
        files: list[str] = []
        keep: list[str] = []
        for i, segment in enumerate(segments):
            local = [href for href in re.findall(r'<a href="([^"]+)"', segment)
                     if "://" not in href and not href.startswith("#")]
            if local:
                files.extend(local)
                # "1 priložena datoteka." — the count line Google writes above the
                # list. Dropped by shape (short, starts with a digit), not by wording.
                if i and keep and re.fullmatch(r"\s*\d+[^<>]{0,40}", keep[-1] or ""):
                    keep.pop()
                continue
            keep.append(segment)

        head, body = (keep[0] if keep else ""), "<br>".join(keep[1:])
        for separator in LABEL_SEPARATORS:
            if separator in head:
                label, _, subject = head.partition(separator)
                return to_text(label).strip(), subject, body, files
        # No separator: the whole line is the label and it names nothing.
        return to_text(head).strip(), "", body, files

    # -- one session -------------------------------------------------------

    def _session(self, reader: _Reader, member: str, digest: str, key: str,
                 cells: list[_Cell], stats: ParseStats) -> Session | None:
        messages: list[Message] = []
        activities: dict[str, int] = {}
        seq = 0
        seen_at: dict[int, int] = {}

        for cell in cells:
            if cell.label:
                activities[cell.label] = activities.get(cell.label, 0) + 1
            # Ids come from the timestamp, not the position: My Activity trims its
            # oldest cells on a retention window, which would renumber everything.
            nth = seen_at.get(cell.at, 0)
            seen_at[cell.at] = nth + 1
            stem = f"{cell.at}-{nth}"

            for message in self._messages(reader, member, cell, stem, stats):
                message.seq = seq
                messages.append(message)
                stats.messages += 1
                stats.parts += len(message.parts)
                seq += 1

        if not messages:
            # Only marker activities ("the Assistant feature was used") — real records,
            # but nothing to read. Counted so the total still reconciles with the file.
            stats.unknown(f"{self.kind}:thread-without-content")
            return None

        title, title_source = self._title(messages, cells)
        conv_id = cells[0].conv_id

        meta = {
            "activities": dict(sorted(activities.items(), key=lambda kv: -kv[1])),
            "attachments": sum(len(c.attachments) for c in cells),
            "turns": len(cells),
            "tz": sorted({c.tz for c in cells}),
            # Point 3: the export never says which Gemini answered, so the model
            # columns stay empty and the app is named as the participant instead.
            "model_recorded": False,
            "participant": self.kind,
            "participant_label": self.label,
            "export_file": reader.path.name,
            "export_member": member.rsplit("/", 1)[-1],
            "export_digest": digest[:16],
        }
        if conv_id:
            meta["conversation_url"] = f"https://gemini.google.com/app/{conv_id}"
        else:
            # Point 5: a canvas document, which the log files under no thread at all.
            meta["unlinked"] = True

        return Session(
            source_kind=self.kind,
            native_id=conv_id or key,
            title=title,
            title_source=title_source,
            started_at=cells[0].at,
            ended_at=cells[-1].at,
            raw_path=str(reader.path),
            exported_at=taken_at(reader.path),
            # Hash this thread's cells, not the document: the export is account-wide,
            # so hashing the file would report all 84 threads as changed whenever one
            # of them gained a turn.
            raw_hash=hashlib.sha256(
                "\x00".join(c.raw for c in cells).encode("utf-8", "replace")
            ).hexdigest(),
            model_primary=None,           # never recorded — see point 3
            messages=messages,            # no token counts and no cost in this export
            meta=meta,
        )

    def _messages(self, reader: _Reader, member: str, cell: _Cell, stem: str,
                  stats: ParseStats) -> list[Message]:
        built: list[Message] = []

        if cell.is_prompt:
            message = Message(native_id=f"{stem}:user", role="user", created_at=cell.at)
            if cell.prompt:
                message.parts.append(self._offload(
                    Part(kind=KIND_TEXT, seq=0, text=cell.prompt,
                         embed_eligible=True), stats))
            for name in cell.attachments:
                part = self._attachment(reader, member, name, len(message.parts), stats)
                if part is not None:
                    message.parts.append(part)
            if message.parts:
                built.append(message)

        if cell.reply:
            # A cell with no prompt but a payload is a canvas document: model output
            # filed under an activity label rather than under a turn.
            message = Message(native_id=f"{stem}:model", role="assistant",
                              created_at=cell.at)
            message.parts.append(self._offload(
                Part(kind=KIND_TEXT, seq=0, text=cell.reply, embed_eligible=True),
                stats))
            if not cell.is_prompt:
                message.meta["activity"] = cell.label
            built.append(message)

        if len(built) > 1:
            for message in built[1:]:
                message.parent_native_id = built[0].native_id
        return built

    def _attachment(self, reader: _Reader, member: str, name: str, seq: int,
                    stats: ParseStats) -> Part | None:
        """A file you uploaded, stored for real — point 6."""
        data = reader.sibling(member, name)
        if data is None:
            # The activity log referenced it, the export did not ship it.
            stats.unknown(f"{self.kind}:attachment-missing")
            return None
        suffix = Path(name).suffix.lower()
        part = Part(kind=KIND_IMAGE if suffix in IMAGE_SUFFIXES else KIND_ATTACHMENT,
                    seq=seq, text=name, bytes=len(data), embed_eligible=False)
        if self.blobs is not None:
            sha, size, dest = self.blobs.put_bytes(data)
            part.blob_sha, part.blob_path, part.bytes = sha, dest, size
            stats.blobs += 1
            stats.blob_bytes += size
        return part

    def _offload(self, part: Part, stats: ParseStats | None = None) -> Part:
        """Oversized parts go to the blob store; a head excerpt stays inline."""
        if part.text and len(part.text) > INLINE_LIMIT and self.blobs is not None:
            stored = self.blobs.put_text(part.text)
            if stored:
                sha, size, dest = stored
                part.blob_sha, part.blob_path, part.bytes = sha, dest, size
                if stats is not None:
                    stats.blobs += 1
                    stats.blob_bytes += size
                part.text = part.text[:INLINE_LIMIT] + "\n…<truncated, full text in blob>"
        return part

    @staticmethod
    def _title(messages: list[Message], cells: list[_Cell]) -> tuple[str | None, str]:
        """There are no titles in this export — the first prompt has to serve as one.

        A canvas has no prompt, but it does have a name, which the NBSP boundary pulls
        out of the localised label: "Ustvarjeno platno Gemini z naslovom<NBSP>Deep
        Learning Masters Quiz App" titles that session *Deep Learning Masters Quiz App*.
        An unnamed canvas falls back to its own opening line, and only a session with no
        readable text at all is reduced to the activity label.
        """
        def first_text(role: str | None) -> str | None:
            return next((p.text for m in messages
                         if role is None or m.role == role
                         for p in m.parts if p.kind == KIND_TEXT and p.text), None)

        text, source = first_text("user"), "first_prompt"
        if not text and cells[0].subject:
            text, source = cells[0].subject, "provider"
        if not text:
            text, source = first_text(None) or cells[0].label, "generated"
        clean = " ".join((text or "").split())
        if not clean:
            return None, source
        if len(clean) > 80:
            clean = (clean[:80].rsplit(" ", 1)[0] or clean[:80]) + "…"
        return clean, source


# -- HTML -> text ----------------------------------------------------------------

class _Markdown(HTMLParser):
    """Render a MyActivity fragment back to the markdown Gemini answered in.

    The log stores answers as rendered HTML — the fences, headings and tables the model
    wrote are gone by the time they reach the file. Keeping the tags would put
    `<strong>` into the FTS index and into every embedding, so they are turned back
    into the markdown they came from rather than stripped to bare prose.
    """

    _HEADINGS = {f"h{n}": n for n in range(1, 7)}
    _WRAPPERS = {"strong": "**", "b": "**", "em": "*", "i": "*"}
    _BLOCKS = ("p", "div", "table", "blockquote")

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self._pre = 0
        self._lists: list[int | None] = []
        self._href: str | None = None
        self._link: list[str] = []
        self._cells = 0

    # `handle_data` is the only place text enters, so link capture redirects here.
    def handle_data(self, data: str) -> None:
        if self._pre:
            self._emit(data, raw=True)
            return
        # Line breaks and indentation survive. Canvas documents are markdown pasted
        # straight into the div — fences, blank lines and indented code, with no <pre>
        # around any of it — so collapsing whitespace the way an HTML renderer does
        # would flatten a source file into one unreadable line.
        text = self._normalise(data.replace("\xa0", " "))
        if text.strip():
            self._emit(text)
        elif "\n" in text and not self._pending_marker():
            self._emit("\n")
        elif text and self.out and not self.out[-1].endswith(("\n", " ")):
            self._emit(" ")

    def _normalise(self, data: str) -> str:
        """Collapse runs of spaces inside a line, but keep every line's indentation."""
        at_start = bool(self.out) and self.out[-1].endswith("\n")
        lines = []
        for i, line in enumerate(data.split("\n")):
            indent = re.match(r"[ \t]*", line).group(0) if (i or at_start) else ""
            lines.append(indent + re.sub(r"[ \t]+", " ", line[len(indent):]))
        return "\n".join(lines)

    def handle_starttag(self, tag: str, attrs) -> None:
        values = dict(attrs)
        if tag == "br":
            self._emit("\n", raw=True)
        elif tag in self._BLOCKS:
            self._block()
        elif tag in self._HEADINGS:
            self._block()
            self._emit("#" * self._HEADINGS[tag] + " ")
        elif tag in ("ul", "ol"):
            self._block()
            self._lists.append(1 if tag == "ol" else None)
        elif tag == "li":
            self._emit("\n", raw=True)
            depth = max(len(self._lists) - 1, 0)
            marker = "- "
            if self._lists and self._lists[-1] is not None:
                marker = f"{self._lists[-1]}. "
                self._lists[-1] += 1
            self._emit("  " * depth + marker)
        elif tag == "pre":
            self._block()
            self._emit("```\n", raw=True)
            self._pre += 1
        elif tag == "code" and not self._pre:
            self._emit("`")
        elif tag in self._WRAPPERS:
            self._emit(self._WRAPPERS[tag])
        elif tag == "hr":
            self._block()
            self._emit("---")
            self._block()
        elif tag == "tr":
            self._emit("\n", raw=True)
            self._cells = 0
        elif tag in ("td", "th"):
            self._emit(" | " if self._cells else "")
            self._cells += 1
        elif tag == "a":
            self._href, self._link = values.get("href"), []
        elif tag == "img":
            self._emit(f"[image: {values.get('src') or '?'}]")

    def handle_startendtag(self, tag: str, attrs) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            text = "".join(self._link).strip()
            href, self._href, self._link = self._href, None, []
            if href and href.startswith("http") and href != text:
                self._emit(f"{text} ({href})" if text else href)
            elif text:
                self._emit(text)
        elif tag == "pre":
            self._pre = max(self._pre - 1, 0)
            # The block's own trailing newline closes the fence; adding another would
            # leave a blank line inside every code block in the archive.
            fence = "```\n" if self.out and self.out[-1].endswith("\n") else "\n```\n"
            self._emit(fence, raw=True)
        elif tag == "code" and not self._pre:
            self._emit("`")
        elif tag in self._WRAPPERS:
            self._emit(self._WRAPPERS[tag])
        elif tag in ("ul", "ol"):
            if self._lists:
                self._lists.pop()
            self._block()
        elif tag in self._BLOCKS or tag in self._HEADINGS:
            self._block()

    def _emit(self, text: str, raw: bool = False) -> None:
        if self._href is not None and not raw:
            self._link.append(text)
            return
        self.out.append(text)

    _PENDING_MARKER = re.compile(r"(?:^|\n)[ \t]*(?:[-*]|\d+\.|#{1,6})[ \t]*$")

    def _pending_marker(self) -> bool:
        """True when the tail is a bullet or heading that has nothing written on it yet.

        Gemini nests a `<p>` inside every `<li>`, and the source newline between the
        two tags arrives here as data. Either would break the line straight after the
        marker, stranding the bullet and pushing its own text into the next paragraph.
        """
        return bool(self._PENDING_MARKER.search("".join(self.out[-3:])))

    def _block(self) -> None:
        """A paragraph break, unless one is already there or a list item just opened."""
        if not self.out or "".join(self.out[-2:]).endswith("\n\n"):
            return
        if self._pending_marker():
            return
        self.out.append("\n\n")

    def result(self) -> str:
        text = "".join(self.out)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def to_text(fragment: str) -> str:
    """HTML fragment -> the markdown that is stored, indexed and embedded."""
    if not fragment or not fragment.strip():
        return ""
    parser = _Markdown()
    try:
        parser.feed(fragment)
        parser.close()
    except Exception:                 # a malformed fragment is content, not a crash
        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", fragment)).strip()
    return parser.result()
