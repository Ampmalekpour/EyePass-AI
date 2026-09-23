"""
gallery.py (recognizer) — NEW, add-face
--------------------------------------------------------------------
Pure `brieface.db` business logic: allocate the next free `c<N>.jpg`
range for a new person and insert their row. Deliberately has no idea
about Redis, MinIO, or the recognition model — same reason
`alignment.py`/`recognition_engine.py` are split out of `worker.py`:
this is the one piece worth unit-testing without spinning up any of
that (see tests/test_gallery.py).

Range allocation strategy — a persisted monotonic counter, NOT
`MAX(range_end)` over live rows:

    The first version of this module computed the next range as
    `MAX(range_end)` across `brieface`. That is wrong the moment a
    person can ever be removed: deleting the highest-numbered row
    drops it out of the MAX, and the next enrollment would silently
    reuse that person's old `c<N>.jpg` numbers — which may still be
    sitting in MinIO (nothing here deletes gallery images), so the new
    person's images would coexist with a dead person's images under
    the same range, and `find_person()`'s range lookup would misattribute
    every match in that block. A tiny `brieface_range_seq` table with a
    single row (`next_start`) is the standard fix: allocation reads and
    advances that counter, independent of what `brieface` itself
    currently contains. It self-bootstraps from `MAX(range_end)` the
    first time it's touched against an existing (pre-this-feature)
    database, so a real deployment's history isn't reset to 1.

Half-open convention (unchanged from the reference): a person owns
image numbers `[range_start, range_end)` — matches the original SQL,
`WHERE range_start <= ? AND range_end > ?`, and `image_filenames()`
below.
--------------------------------------------------------------------
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import List, Tuple

_SEQ_TABLE = "brieface_range_seq"


@dataclass(frozen=True)
class PersonRecord:
    name: str
    lastname: str
    section: str
    codeid: str
    personnelid: str


def _ensure_seq_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {_SEQ_TABLE} (id INTEGER PRIMARY KEY CHECK (id = 1), next_start INTEGER NOT NULL)"
    )
    row = conn.execute(f"SELECT next_start FROM {_SEQ_TABLE} WHERE id = 1").fetchone()
    if row is not None:
        return
    # First time this db is touched by add-face: bootstrap the counter
    # from whatever's already there (an existing gallery migrated from
    # the reference system, or a brand-new empty one) rather than
    # resetting history to 1.
    (seed,) = conn.execute("SELECT COALESCE(MAX(range_end), 1) FROM brieface").fetchone()
    conn.execute(f"INSERT INTO {_SEQ_TABLE} (id, next_start) VALUES (1, ?)", (int(seed),))
    conn.commit()


def next_free_range(conn: sqlite3.Connection, count: int) -> Tuple[int, int]:
    """Returns (range_start, range_end) for `count` new images WITHOUT
    advancing the counter — read-only preview. `allocate_and_insert()`
    is what actually commits the advance; call this directly only for
    inspection (e.g. tests, an admin "what would the next range be"
    check)."""
    _ensure_seq_table(conn)
    (range_start,) = conn.execute(f"SELECT next_start FROM {_SEQ_TABLE} WHERE id = 1").fetchone()
    return int(range_start), int(range_start) + int(count)


def allocate_and_insert(conn: sqlite3.Connection, person: PersonRecord, image_count: int) -> Tuple[int, int]:
    """Allocates the next free range, advances the persisted counter,
    and inserts the person's row — all in one transaction. Returns
    (range_start, range_end); the caller (worker.py) uses range_start
    to name the incoming c<N>.jpg files.

    Caller is responsible for holding bus.gallery_lock() around this
    AND around the subsequent file writes / MinIO upload / commit —
    the lock's job is to make "allocate range" and "the range actually
    becomes occupied on disk" atomic as a whole, not just this insert.
    (The counter advance here is safe even without the lock — SQLite
    serializes writers to one file — but the lock is still required to
    stop two concurrent commits from racing on the MinIO upload step,
    which SQLite's own locking has no say over.)
    """
    range_start, range_end = next_free_range(conn, image_count)
    conn.execute(f"UPDATE {_SEQ_TABLE} SET next_start = ? WHERE id = 1", (range_end,))
    conn.execute(
        """
        INSERT INTO brieface (name, lastname, section, codeid, personnelid, range_start, range_end)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (person.name, person.lastname, person.section, person.codeid, person.personnelid,
         range_start, range_end),
    )
    conn.commit()
    return range_start, range_end


def image_filenames(range_start: int, range_end: int) -> List[str]:
    """c<N>.jpg for each N in the allocated range — matches the naming
    convention `recognition_engine.py::_decide_person_and_confidence`
    already assumes when it maps a matched filename back to a person
    (`(image_num - 1) // 3 + 1`, i.e. a fixed, equal-sized block per
    person)."""
    return [f"c{n}.jpg" for n in range(range_start, range_end)]
