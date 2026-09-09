"""Topic groups derived from the embeddings, so that grouping exists without being typed.

Manual tagging has been in the web UI since the first week and the `tag` table has never
held a row. That is the argument for this module: the only grouping that will actually
exist is one nobody has to maintain. A session is already a point in embedding space —
mean-pool its chunk vectors and sessions about the same thing land near each other.

Three deliberate choices:

**Average linkage, not k-means.** k is a guess that has to be re-guessed as the archive
grows, and every session gets forced into a group whether or not it belongs to one. A
similarity threshold is a knob that means something you can hold in your head -- "how alike
must two sessions be to file together" -- and the leftovers stay leftovers.

**No sklearn.** For unit vectors the average-linkage similarity between two clusters is
exactly the dot product of their SUM vectors over the product of their sizes, because
mean(a.b) over all pairs == (SUM_A . SUM_B) / (|A||B|). So the whole algorithm is sum
vectors and one matmul per merge, in the same spirit as `embed.py` choosing a brute-force
matmul over FAISS.

**c-TF-IDF for labels.** A group whose label is "cluster 7" is not a browse facet. Scoring
each group's own words against the other groups' is the cheap standard trick, and it needs
nothing but `collections.Counter`.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
import unicodedata
from collections import Counter, defaultdict

import numpy as np

from .embed import MODEL_TAG, VectorStore
from .hybrid import pool_rows

# How alike two groups must be to merge. Chosen by eye against this archive: below ~0.45
# unrelated projects fold together, above ~0.65 nothing merges at all and every session is
# its own group. Exposed as `llma topics --threshold` because it is a judgment about your
# own material, not a constant.
DEFAULT_THRESHOLD = 0.55

# A group of two is a coincidence, not a topic. Smaller groups are dissolved and their
# sessions read as unclustered, which is honest -- most of a personal archive is one-off
# questions that belong to nothing.
DEFAULT_MIN_SIZE = 3

TERMS_IN_LABEL = 3

# Words that carry no topic. English and Slovene together: ~14% of this archive is Slovene,
# which is also why the FTS tokenizer runs `remove_diacritics 2`. Without the Slovene half
# every group here labelled itself "kako", "lahko", "kaj".
_STOPWORDS = {
    # English
    "the", "and", "for", "you", "your", "not", "but", "with", "this", "that", "have",
    "from", "are", "was", "were", "can", "will", "would", "should", "could", "how",
    "what", "when", "where", "which", "who", "why", "all", "any", "get", "got", "has",
    "had", "her", "his", "its", "our", "out", "too", "use", "used", "using", "way",
    "one", "two", "new", "now", "see", "make", "made", "does", "did", "doing", "just",
    "like", "want", "need", "know", "think", "into", "over", "than", "then", "them",
    "they", "there", "here", "some", "more", "most", "other", "also", "only", "very",
    "same", "such", "these", "those", "been", "being", "about", "after", "before",
    "because", "while", "still", "much", "many", "each", "both", "own", "off", "why",
    "add", "set", "run", "try", "let", "put", "via", "per", "yes",
    # Slovene
    "kako", "lahko", "kaj", "kje", "kdaj", "zakaj", "kateri", "katera", "katero",
    "sem", "smo", "ste", "sta", "bom", "bos", "bomo", "bodo", "bil", "bila", "bilo",
    "bili", "biti", "jih", "jim", "jaz", "kar", "ker", "kot", "med", "mora", "moram",
    "nad", "naj", "narediti", "nato", "nekaj", "niso", "pod", "pol", "potem", "prav",
    "pred", "pri", "samo", "sedaj", "tako", "tam", "tega", "temu", "tisti", "tudi",
    "vam", "vas", "vec", "ves", "vse", "vsi", "zdaj", "zelo", "kjer", "ali", "pa",
    "ampak", "torej", "hvala", "prosim", "dobro", "treba", "moje", "moja", "moj",
    # cross-language noise that survives both lists
    "https", "http", "com", "org", "www", "file", "files", "code", "line", "lines",
    "error", "errors", "session", "sessions", "chat", "claude", "gpt", "assistant",
    "user", "please", "help", "thanks", "okay",
}

_WORD_RE = re.compile(r"[^\W\d_]{3,}", re.UNICODE)


def _fold(word: str) -> str:
    """Casefold and strip diacritics, so `crke`/`crke` count as one term.

    Matches what the FTS tokenizer does with `remove_diacritics 2`. Only the COUNTING key
    is folded -- the label keeps whichever surface form was most common, so a Slovene group
    still reads in Slovene.
    """
    stripped = unicodedata.normalize("NFKD", word.casefold())
    return "".join(c for c in stripped if not unicodedata.combining(c))


def session_vectors(con, vectors_dir, model_tag: str = MODEL_TAG):
    """(session_ids, unit vectors) for every session that has chunks. One pass, one query.

    Not `hybrid._session_centroid` in a loop: that is one query per session, and it would
    load the vector file 550 times. The pooling itself is the shared `pool_rows`, so a
    session's vector here is the same vector its Related panel is built from.
    """
    store = VectorStore(vectors_dir, model_tag)
    matrix = store.load()
    if matrix is None or len(matrix) == 0:
        return [], np.zeros((0, 0), dtype=np.float32)

    rows_by_session: dict[int, list[int]] = defaultdict(list)
    for row in con.execute(
            "SELECT session_id, vec_row FROM chunk WHERE model_tag = ? "
            "ORDER BY session_id, seq", (model_tag,)):
        if 0 <= row["vec_row"] < len(matrix):
            rows_by_session[row["session_id"]].append(row["vec_row"])

    ids, vecs = [], []
    for session_id in sorted(rows_by_session):
        pooled = pool_rows(matrix, rows_by_session[session_id])
        if pooled is not None:
            ids.append(session_id)
            vecs.append(pooled)
    if not vecs:
        return [], np.zeros((0, 0), dtype=np.float32)
    return ids, np.vstack(vecs).astype(np.float32)


def cluster(vecs: np.ndarray, threshold: float = DEFAULT_THRESHOLD,
            min_size: int = DEFAULT_MIN_SIZE) -> list[list[int]]:
    """Agglomerative average-linkage over unit vectors. Returns groups of row indices.

    Exact average linkage, not an approximation of it: for unit vectors the mean pairwise
    cosine between two clusters is (SUM_A . SUM_B) / (|A| |B|), so carrying sum vectors
    gives the true linkage with one dot product per candidate pair and no distance matrix
    to update. Groups come back largest first; anything under `min_size` is dropped, and
    its sessions are simply not assigned.
    """
    n = len(vecs)
    if n == 0:
        return []

    sums = vecs.astype(np.float32).copy()       # per-cluster sum vector
    sizes = np.ones(n, dtype=np.float32)
    members: list[list[int]] = [[i] for i in range(n)]
    alive = np.ones(n, dtype=bool)

    while True:
        live = np.flatnonzero(alive)
        if len(live) < 2:
            break
        # average linkage for every live pair at once
        sim = (sums[live] @ sums[live].T) / np.outer(sizes[live], sizes[live])
        np.fill_diagonal(sim, -np.inf)
        flat = int(np.argmax(sim))
        best = sim.flat[flat]
        if best < threshold:
            break
        i, j = divmod(flat, len(live))
        a, b = int(live[i]), int(live[j])
        sums[a] += sums[b]
        sizes[a] += sizes[b]
        members[a].extend(members[b])
        alive[b] = False
        members[b] = []

    groups = [members[i] for i in np.flatnonzero(alive) if len(members[i]) >= min_size]
    # Largest first, then by lowest member index, so a rebuild over unchanged data
    # produces the same order and the facet does not reshuffle for no reason.
    groups.sort(key=lambda g: (-len(g), min(g)))
    return groups


def _documents(con, session_ids: list[int]) -> dict[int, str]:
    """The text each session is labelled from: its title and its opening user turns.

    The same material `hybrid._session_gist` builds a BM25 query from, and for the same
    reason -- a whole transcript would label every group with whatever stack trace was
    pasted into it.
    """
    from .hybrid import _session_gist
    return {sid: _session_gist(con, sid, chars=800) for sid in session_ids}


def label_groups(con, ids: list[int], groups: list[list[int]]) -> list[dict]:
    """Name each group by the words that distinguish it from the others (c-TF-IDF)."""
    docs = _documents(con, [ids[i] for g in groups for i in g])

    counts: list[Counter] = []
    surface: list[dict[str, str]] = []          # folded key -> most common original spelling
    for group in groups:
        counter: Counter = Counter()
        seen: dict[str, Counter] = defaultdict(Counter)
        for row in group:
            for word in _WORD_RE.findall(docs.get(ids[row], "")):
                key = _fold(word)
                if key in _STOPWORDS or len(key) < 3:
                    continue
                counter[key] += 1
                seen[key][word.lower()] += 1
        counts.append(counter)
        surface.append({k: v.most_common(1)[0][0] for k, v in seen.items()})

    # document frequency across groups, not across sessions: the question is which words
    # separate THIS group from the others
    df: Counter = Counter()
    for counter in counts:
        df.update(counter.keys())

    n_groups = max(len(groups), 1)
    out = []
    used: set[str] = set()
    for gi, group in enumerate(groups):
        counter = counts[gi]
        total = sum(counter.values()) or 1
        scored = sorted(
            ((word, (freq / total) * np.log(1 + n_groups / df[word]))
             for word, freq in counter.items()),
            key=lambda kv: -kv[1])
        # Skip terms already spent on an earlier (larger) group, so two neighbouring
        # groups do not both come out as "packet tracer".
        picked = []
        for word, score in scored:
            if word in used:
                continue
            picked.append((word, float(score)))
            if len(picked) == TERMS_IN_LABEL:
                break
        if not picked:                       # every distinguishing word already taken
            picked = [(w, float(s)) for w, s in scored[:TERMS_IN_LABEL]]
        used.update(w for w, _ in picked)

        label = " · ".join(surface[gi].get(w, w) for w, _ in picked) or f"group {gi + 1}"
        out.append({
            "label": label,
            "slug": _slug(label, gi),
            "terms": [{"term": w, "score": round(s, 5)} for w, s in picked],
            "rows": group,
        })
    return out


def _slug(label: str, index: int) -> str:
    folded = _fold(label.replace("·", " "))
    slug = re.sub(r"[^a-z0-9]+", "-", folded).strip("-")
    return slug or f"group-{index + 1}"


def build(con: sqlite3.Connection, vectors_dir, *,
          threshold: float = DEFAULT_THRESHOLD, min_size: int = DEFAULT_MIN_SIZE,
          model_tag: str = MODEL_TAG) -> dict:
    """Recompute every topic group. Replaces what was there; safe to re-run.

    Wholesale rather than incremental on purpose: labels are relative to the other groups,
    so one new session can legitimately rename a group, and a half-updated set of labels
    would be worse than a slightly stale one.
    """
    ids, vecs = session_vectors(con, vectors_dir, model_tag)
    if not ids:
        # Nothing to cluster -- almost always "indexed with --no-vectors". Clearing anyway
        # would silently throw away a good set of groups the moment a keyword-only rebuild
        # ran, which is the same trap `index.py` avoids with its chunks.
        return {"topics": 0, "assigned": 0, "unclustered": 0, "skipped": True,
                "reason": "no session vectors — run `llma index` with vectors first"}

    groups = cluster(vecs, threshold=threshold, min_size=min_size)
    labelled = label_groups(con, ids, groups)

    con.execute("DELETE FROM session_topic")
    con.execute("DELETE FROM topic")
    built_at = int(time.time() * 1000)
    assigned = 0
    for entry in labelled:
        rows = entry["rows"]
        centroid = vecs[rows].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        cur = con.execute(
            "INSERT INTO topic(slug,label,size,terms,built_at,model_tag) "
            "VALUES (?,?,?,?,?,?)",
            (entry["slug"], entry["label"], len(rows),
             json.dumps(entry["terms"], ensure_ascii=False), built_at, model_tag))
        topic_id = cur.lastrowid
        for row in rows:
            con.execute(
                "INSERT INTO session_topic(session_id,topic_id,similarity) VALUES (?,?,?)",
                (ids[row], topic_id, float(vecs[row] @ centroid)))
            assigned += 1
    con.commit()

    return {"topics": len(labelled), "assigned": assigned,
            "unclustered": len(ids) - assigned, "vectors": len(ids),
            "threshold": threshold, "min_size": min_size, "skipped": False,
            "groups": [{"slug": e["slug"], "label": e["label"], "size": len(e["rows"])}
                       for e in labelled]}


def topic_facet(con) -> list[dict]:
    """Groups for the browse/search sidebar, largest first, plus the unclustered count."""
    rows = con.execute("""
        SELECT t.slug, t.label, COUNT(st.session_id) n
        FROM topic t LEFT JOIN session_topic st ON st.topic_id = t.id
        GROUP BY t.id ORDER BY n DESC, t.label""").fetchall()
    return [dict(r) for r in rows]
