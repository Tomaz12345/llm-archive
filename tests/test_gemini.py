"""Regression tests for the Google Takeout "My Activity" adapter.

Fixtures are built from the shapes verified in the 4.8 MB Slovene export (745 cells /
84 threads / 52 canvases): a prompt cell, a prompt with an attached file, a canvas with
no thread link, a marker activity with nothing to read. Those are the four shapes the
adapter makes a decision about, so those are what is pinned here — plus the two rules
that a locale change would otherwise break silently: the NBSP label boundary and the
localised month abbreviation.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from llm_archive.adapters.gemini import GeminiAdapter, to_text
from llm_archive.core.blobs import BlobStore
from llm_archive.core.models import KIND_IMAGE, KIND_TEXT, ParseStats

FOLDER = "Takeout/Moja dejavnost/Aplikacije Gemini"
DOCUMENT = f"{FOLDER}/Moja_dejavnost.html"

PROMPT_LABEL = "Poslali ste poziv:"
CANVAS_LABEL = "Ustvarjeno platno Gemini z naslovom"
MARKER_LABEL = "Uporabljena je bila funkcija Pomočnika"

CAPTION = (
    '<div class="content-cell mdl-cell mdl-cell--12-col mdl-typography--caption">'
    "<b>Izdelki:</b><br>&emsp;Aplikacije Gemini<br>{details}"
    "<b>Zakaj je to tukaj?</b><br>&emsp;Ta dejavnost je bila shranjena v Google "
    "Račun.</div>"
)
DETAILS = ('<b>Podrobnosti:</b><br>&emsp;https://gemini.google.com/app/{cid}: '
           '<a href="https://gemini.google.com/app/{cid}">'
           "https://gemini.google.com/app/{cid}</a><br>")


def cell(activity: str, when: str = "5. nov. 2025, 01:34:47 CEST",
         reply: str = "", cid: str | None = "abc123", preview: str = "",
         product: str = "Aplikacije Gemini") -> str:
    """One `outer-cell`, in the order the export writes it."""
    details = DETAILS.format(cid=cid) if cid else ""
    return (
        '<div class="outer-cell mdl-cell mdl-cell--12-col mdl-shadow--2dp">'
        '<div class="mdl-grid">'
        '<div class="header-cell mdl-cell mdl-cell--12-col">'
        f'<p class="mdl-typography--title">{product}<br></p></div>'
        '<div class="content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1">'
        f"{activity}<br>{when}<br>{reply}</div>"
        '<div class="content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1 '
        f'mdl-typography--text-right">{preview}</div>'
        + CAPTION.format(details=details) + "</div></div>"
    )


def document(*cells: str) -> str:
    return ("<html><head><title>Zgodovina moje dejavnosti</title>"
            "<style>.mdl-grid{display:flex}</style></head>"
            '<body><div class="mdl-grid">' + "".join(cells) + "</div></body></html>")


def export(tmp_path: Path, html: str, files: dict[str, bytes] | None = None) -> Path:
    drops = tmp_path / "drops"
    drops.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(drops / "takeout-20260826T202708Z-1-001.zip", "w") as zf:
        zf.writestr(DOCUMENT, html)
        for name, data in (files or {}).items():
            zf.writestr(f"{FOLDER}/{name}", data)
    return drops


def parse(drops: Path, blobs: BlobStore | None = None) -> tuple[list, ParseStats]:
    adapter = GeminiAdapter(drops=drops, blobs=blobs)
    stats = ParseStats()
    found = adapter.discover()
    return [s for path in found for s in adapter.parse(path, stats)], stats


# -- discovery -------------------------------------------------------------------

def test_discovers_by_markup_not_by_filename(tmp_path):
    """Every name in a Takeout path is localised, so none of them can be matched on."""
    drops = export(tmp_path, document(cell(f"{PROMPT_LABEL}\xa0Hi", reply="<p>Hey</p>")))
    assert [p.name for p in GeminiAdapter(drops=drops).discover()] \
        == ["takeout-20260826T202708Z-1-001.zip"]


def test_ignores_another_products_activity_log(tmp_path):
    """A full Takeout ships a dozen of these; only the Gemini one is ours."""
    drops = tmp_path / "drops"
    drops.mkdir()
    with zipfile.ZipFile(drops / "takeout-search.zip", "w") as zf:
        zf.writestr("Takeout/Moja dejavnost/Iskanje/Moja_dejavnost.html",
                    document(cell("Iskali ste\xa0kranj vreme", cid=None,
                                  product="Iskanje")))
    assert GeminiAdapter(drops=drops).discover() == []


def test_a_non_zip_drop_is_not_an_error(tmp_path):
    drops = tmp_path / "drops"
    drops.mkdir()
    (drops / "notes.txt").write_text("not an export", encoding="utf-8")
    (drops / "threads-export-2026.json").write_text("{}", encoding="utf-8")
    assert GeminiAdapter(drops=drops).discover() == []


# -- the timestamp split ---------------------------------------------------------

def test_prompt_and_reply_split_on_the_timestamp(tmp_path):
    """Point 2: the stamp is the only boundary between you and the model."""
    sessions, stats = parse(export(tmp_path, document(
        cell(f"{PROMPT_LABEL}\xa0How do I rebase?", reply="<p>Use <code>-i</code>.</p>"))))

    assert len(sessions) == 1
    session = sessions[0]
    assert [m.role for m in session.messages] == ["user", "assistant"]
    assert session.messages[0].parts[0].text == "How do I rebase?"
    assert session.messages[1].parts[0].text == "Use `-i`."
    assert session.messages[0].seq == 0 and session.messages[1].seq == 1
    assert stats.messages == 2


def test_a_cell_without_a_timestamp_is_counted_not_guessed(tmp_path):
    """Mis-splitting would file the model's answer under the user's name."""
    _, stats = parse(export(tmp_path, document(
        cell(f"{PROMPT_LABEL}\xa0Hi", when="some time last week", reply="<p>Hey</p>"))))
    assert stats.unknown_types == {"gemini:cell-without-timestamp": 1}


def test_timestamp_is_utc_epoch_ms_from_the_printed_offset(tmp_path):
    """01:34:47 CEST is 23:34:47 UTC the day before — §8.4."""
    sessions, _ = parse(export(tmp_path, document(
        cell(f"{PROMPT_LABEL}\xa0Hi", reply="<p>Hey</p>"))))
    assert sessions[0].started_at == 1762299287000


def test_english_export_parses_too(tmp_path):
    """Same instant, written the way an English-locale export writes it."""
    sessions, _ = parse(export(tmp_path, document(
        cell("You sent a prompt:\xa0Hi", when="Nov 5, 2025, 1:34:47 AM CEST",
             reply="<p>Hey</p>"))))
    assert sessions[0].started_at == 1762299287000
    assert sessions[0].messages[0].parts[0].text == "Hi"


def test_an_unreadable_month_is_reported(tmp_path):
    _, stats = parse(export(tmp_path, document(
        cell(f"{PROMPT_LABEL}\xa0Hi", when="5. brumaire 2025, 01:34:47 CEST",
             reply="<p>Hey</p>"))))
    assert stats.unknown_types == {"gemini:month:brumaire": 1,
                                  "gemini:cell-without-timestamp": 1}


# -- who spoke -------------------------------------------------------------------

def test_canvas_is_the_models_work_not_yours(tmp_path):
    """Point 5: no colon before the NBSP means Gemini produced this, not you."""
    sessions, _ = parse(export(tmp_path, document(
        cell(f"{CANVAS_LABEL}\xa0Quiz App<br>```\nprint(1)\n```", cid=None))))

    session = sessions[0]
    assert [m.role for m in session.messages] == ["assistant"]
    assert session.messages[0].parts[0].text == "```\nprint(1)\n```"
    assert session.title == "Quiz App"          # the NBSP separates it from the label
    assert session.meta["unlinked"] is True
    assert session.native_id.startswith("loose:")
    assert session.messages[0].meta["activity"] == CANVAS_LABEL


def test_a_canvas_keeps_the_same_id_across_exports(tmp_path):
    """Keyed by content, so re-exporting does not clone every canvas."""
    html = document(cell(f"{CANVAS_LABEL}\xa0Quiz App<br>```\nprint(1)\n```", cid=None))
    first, _ = parse(export(tmp_path, html))
    second, _ = parse(export(tmp_path / "again", html))
    assert first[0].native_id == second[0].native_id
    assert first[0].raw_hash == second[0].raw_hash


def test_an_activity_with_nothing_to_read_is_counted(tmp_path):
    """The "Assistant feature was used" marker: a real record, but no content."""
    sessions, stats = parse(export(tmp_path, document(cell(MARKER_LABEL, cid="ghost"))))
    assert sessions == []
    assert stats.unknown_types == {"gemini:thread-without-content": 1}


def test_a_marker_does_not_hide_the_thread_it_belongs_to(tmp_path):
    sessions, _ = parse(export(tmp_path, document(
        cell(MARKER_LABEL, when="5. nov. 2025, 01:35:00 CEST"),
        cell(f"{PROMPT_LABEL}\xa0Hi", reply="<p>Hey</p>"))))

    assert len(sessions) == 1
    assert [m.role for m in sessions[0].messages] == ["user", "assistant"]
    assert sessions[0].meta["turns"] == 2        # the marker still bounds the session
    assert sessions[0].ended_at == 1762299300000
    assert sessions[0].meta["activities"][MARKER_LABEL] == 1


# -- threads ---------------------------------------------------------------------

def test_cells_are_regrouped_into_threads_oldest_first(tmp_path):
    """Point 1: an activity log is newest-first and scattered; a thread is neither."""
    sessions, _ = parse(export(tmp_path, document(
        cell(f"{PROMPT_LABEL}\xa0Third", when="7. nov. 2025, 01:00:00 CEST",
             reply="<p>C</p>", cid="one"),
        cell(f"{PROMPT_LABEL}\xa0Other", when="6. nov. 2025, 01:00:00 CEST",
             reply="<p>X</p>", cid="two"),
        cell(f"{PROMPT_LABEL}\xa0First", when="5. nov. 2025, 01:00:00 CEST",
             reply="<p>A</p>", cid="one"))))

    threads = {s.native_id: s for s in sessions}
    assert set(threads) == {"one", "two"}
    first = threads["one"]
    assert [p.text for m in first.messages for p in m.parts] == \
        ["First", "A", "Third", "C"]
    assert first.title == "First"                # the oldest prompt titles the thread
    assert first.title_source == "first_prompt"
    assert first.started_at < first.ended_at
    assert first.meta["conversation_url"] == "https://gemini.google.com/app/one"


def test_same_second_ties_break_on_document_order(tmp_path):
    """Two cells in one second: the later one in the file is the earlier turn."""
    sessions, _ = parse(export(tmp_path, document(
        cell(f"{PROMPT_LABEL}\xa0Second", reply="<p>B</p>"),
        cell(f"{PROMPT_LABEL}\xa0First", reply="<p>A</p>"))))
    assert [p.text for m in sessions[0].messages for p in m.parts] == \
        ["First", "A", "Second", "B"]
    assert len({m.native_id for m in sessions[0].messages}) == 4


def test_ids_survive_the_retention_window_trimming_old_cells(tmp_path):
    """My Activity deletes its oldest cells; positional ids would renumber the rest."""
    full, _ = parse(export(tmp_path, document(
        cell(f"{PROMPT_LABEL}\xa0Newer", when="7. nov. 2025, 01:00:00 CEST",
             reply="<p>B</p>"),
        cell(f"{PROMPT_LABEL}\xa0Older", when="5. nov. 2025, 01:00:00 CEST",
             reply="<p>A</p>"))))
    trimmed, _ = parse(export(tmp_path / "later", document(
        cell(f"{PROMPT_LABEL}\xa0Newer", when="7. nov. 2025, 01:00:00 CEST",
             reply="<p>B</p>"))))

    kept = {m.native_id for m in trimmed[0].messages}
    assert kept <= {m.native_id for m in full[0].messages}
    assert trimmed[0].native_id == full[0].native_id


def test_no_model_and_no_usage_are_left_empty(tmp_path):
    """Point 3: the export records none of these, so none of them are invented."""
    sessions, _ = parse(export(tmp_path, document(
        cell(f"{PROMPT_LABEL}\xa0Hi", reply="<p>Hey</p>"))))
    session = sessions[0]
    assert session.model_primary is None
    assert (session.tok_in, session.tok_out, session.cost_usd) == (None, None, None)
    assert all(m.model is None for m in session.messages)
    assert session.meta["model_recorded"] is False
    assert session.meta["participant"] == "gemini"
    assert session.meta["tz"] == ["CEST"]


# -- attachments -----------------------------------------------------------------

def test_attachment_bytes_are_stored(tmp_path):
    """Point 6: this is the one export that actually ships the files you uploaded."""
    png = b"\x89PNG\r\n\x1a\n" + b"pixels" * 40
    html = document(cell(
        f"{PROMPT_LABEL}\xa0What is wrong here?<br>1 priložena datoteka.<br>"
        '-  <a href="slika-52785bee74b1e9bd.png">slika.png</a>',
        reply="<p>Port forwarding.</p>",
        preview='<img src="slika-52785bee74b1e9bd.png" class="image-preview"><br>'))
    blobs = BlobStore(tmp_path / "blobs")
    sessions, stats = parse(export(tmp_path, html,
                                   {"slika-52785bee74b1e9bd.png": png}), blobs)

    prompt = sessions[0].messages[0]
    assert [p.kind for p in prompt.parts] == [KIND_TEXT, KIND_IMAGE]
    # The count line is Google's, not yours: it does not belong in the prompt.
    assert prompt.parts[0].text == "What is wrong here?"
    image = prompt.parts[1]
    assert image.text == "slika-52785bee74b1e9bd.png"
    assert image.bytes == len(png) and image.embed_eligible is False
    assert Path(image.blob_path).read_bytes() == png
    assert stats.blobs == 1 and sessions[0].meta["attachments"] == 1


def test_a_referenced_file_the_export_omitted_is_reported(tmp_path):
    sessions, stats = parse(export(tmp_path, document(cell(
        f"{PROMPT_LABEL}\xa0Look<br>1 priložena datoteka.<br>"
        '-  <a href="slika-gone.png">slika.png</a>', reply="<p>Sure.</p>"))))
    assert stats.unknown_types == {"gemini:attachment-missing": 1}
    assert [p.kind for p in sessions[0].messages[0].parts] == [KIND_TEXT]


# -- HTML -> markdown ------------------------------------------------------------

def test_rendered_html_becomes_the_markdown_it_came_from():
    """`<strong>` in the FTS index and in every embedding is the thing to avoid."""
    assert to_text("<p>Use <strong>this</strong> and <em>that</em>.</p>") == \
        "Use **this** and *that*."
    assert to_text("<h2>Steps</h2><ul><li><p>First</p></li><li><p>Second</p></li></ul>") \
        == "## Steps\n\n- First\n\n- Second"
    assert to_text("<ol><li>One</li><li>Two</li></ol>") == "1. One\n2. Two"
    assert to_text("<pre><code>x = 1\ny = 2\n</code></pre>") == "```\nx = 1\ny = 2\n```"
    assert to_text("<p>a &lt; b &amp;&amp; c &gt; d</p>") == "a < b && c > d"
    assert to_text("<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr>"
                   "</table>") == "A | B\n1 | 2"


def test_canvas_indentation_is_not_collapsed():
    """A canvas is markdown in a bare div — no <pre> to protect its whitespace."""
    assert to_text("def f():\n    if x:\n        return  1\n") == \
        "def f():\n    if x:\n        return 1"


def test_a_link_keeps_its_target():
    assert to_text('<p>See <a href="https://example.com/x">the docs</a>.</p>') == \
        "See the docs (https://example.com/x)."


def test_malformed_markup_degrades_to_text_rather_than_raising():
    assert "hello" in to_text("<p>hello<<<>>")
