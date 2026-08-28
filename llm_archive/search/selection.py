"""What counts as embeddable — defined once, for the indexer and for the health check.

These two must agree exactly. When they did not, two parts hanging off orphaned message
rows (a session row that no longer existed) were invisible to the indexer's `JOIN
session` but visible to a health query that joined only `part -> message`. The result
was a permanent "2 embeddable parts not in the vector index" on a freshly built index —
a warning that can never be cleared teaches you to ignore the warning.

Kept dependency-free on purpose: `index` pulls in numpy and the embedding model, and
`/stats` must not pay for that just to count rows.
"""

from __future__ import annotations

# The join chain is part of the definition, not decoration: a part whose message or
# session has gone is not embeddable, because the indexer cannot reach it.
EMBEDDABLE_FROM = """
    FROM part p
    JOIN message m ON m.id = p.message_id
    JOIN session s ON s.id = m.session_id
    JOIN source src ON src.id = s.source_id
    LEFT JOIN workspace w ON w.id = s.workspace_id
"""

EMBEDDABLE_WHERE = """
    WHERE p.embed_eligible = 1
      AND p.text IS NOT NULL
      AND LENGTH(p.text) >= 40
      AND m.on_active_path = 1
"""


def unindexed_count(con, model_tag: str | None = None) -> int:
    """Embeddable parts with no chunk — text vector search cannot reach.

    Zero on a fresh build. Anything else means the index no longer covers the archive.
    """
    extra = " AND c.model_tag = ?" if model_tag else ""
    params = (model_tag,) if model_tag else ()
    return con.execute(f"""
        SELECT COUNT(*) {EMBEDDABLE_FROM} {EMBEDDABLE_WHERE}
          AND NOT EXISTS (SELECT 1 FROM chunk c
                          WHERE c.part_id = p.id{extra})
    """, params).fetchone()[0]
