"""Active-path resolution for tree-structured transcripts.

Claude Code, Claude.ai and ChatGPT all store a DAG rather than a list: editing a prompt
or rewinding leaves the abandoned turns in the file. Reading them in document order
produces the same question answered several ways, presented as one conversation.

Shared here because all three need identical logic, and getting it subtly wrong is
silent — it corrupts search and statistics without ever raising an error.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Hashable, Iterable


def resolve_active_path(
    ids: Iterable[Hashable],
    parent_of: Callable[[Hashable], Hashable | None],
    sort_key: Callable[[Hashable], object],
) -> set:
    """Return the ids on the surviving root-to-leaf path.

    `ids` must include EVERY node that can appear in a parent link, not just the
    conversational ones. Claude Code chains hop through bookkeeping records, so
    filtering before building the graph shatters one tree into dozens of false roots
    and collapses the active path to a single node.

    The newest leaf wins: whatever was written last is what the conversation became.
    """
    nodes = set(ids)
    if not nodes:
        return set()

    children: dict[Hashable, list] = defaultdict(list)
    for node in nodes:
        parent = parent_of(node)
        if parent in nodes:
            children[parent].append(node)

    leaves = [n for n in nodes if not children.get(n)]
    if not leaves:
        return set(nodes)          # a cycle; keep everything rather than lose it

    cursor = max(leaves, key=sort_key)
    active: set = set()
    while cursor is not None and cursor in nodes and cursor not in active:
        active.add(cursor)
        cursor = parent_of(cursor)
    return active
