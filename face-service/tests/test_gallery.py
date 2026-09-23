"""
test_gallery.py
--------------------------------------------------------------------
Exercises recognizer/src/gallery.py's range-allocation logic directly
against an in-memory sqlite db — no Redis, no torch, no MinIO. Mirrors
tests/test_engine_manager_rebalance.py's "real logic, fake plumbing"
approach.

Run with:
    PYTHONPATH=recognizer/src python3 tests/test_gallery.py
--------------------------------------------------------------------
"""

import os
import sqlite3
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "recognizer", "src"))

import gallery  # noqa: E402


def _make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE brieface (
            name TEXT, lastname TEXT, section TEXT, codeid TEXT,
            personnelid TEXT, range_start INTEGER, range_end INTEGER
        )
    """)
    conn.commit()
    return conn


class NextFreeRangeTests(unittest.TestCase):
    def test_empty_table_starts_at_one(self):
        conn = _make_db()
        start, end = gallery.next_free_range(conn, count=3)
        self.assertEqual((start, end), (1, 4))

    def test_bootstraps_from_existing_rows_on_first_touch(self):
        # Simulates pointing this code at a gallery migrated from the
        # reference system: rows already exist, but the sequence table
        # (introduced by this revision) does not yet. First read must
        # seed itself from the highest range_end already in use, not
        # start over at 1 and collide with existing images.
        conn = _make_db()
        conn.execute(
            "INSERT INTO brieface VALUES (?,?,?,?,?,?,?)",
            ("Ali", "Rezaei", "IT", "0011", "1001", 1, 4),
        )
        conn.commit()
        start, end = gallery.next_free_range(conn, count=3)
        self.assertEqual((start, end), (4, 7))


class AllocateAndInsertTests(unittest.TestCase):
    def test_inserts_row_and_returns_matching_range(self):
        conn = _make_db()
        person = gallery.PersonRecord(name="Sara", lastname="Ahmadi", section="HR",
                                       codeid="0099", personnelid="2001")
        start, end = gallery.allocate_and_insert(conn, person, image_count=3)
        self.assertEqual((start, end), (1, 4))

        row = conn.execute("SELECT * FROM brieface WHERE personnelid = ?", (person.personnelid,)).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[5], start)
        self.assertEqual(row[6], end)

    def test_two_sequential_enrollments_never_collide(self):
        conn = _make_db()
        p1 = gallery.PersonRecord("A", "A", "S", "0", "1")
        p2 = gallery.PersonRecord("B", "B", "S", "0", "2")
        r1 = gallery.allocate_and_insert(conn, p1, image_count=3)
        r2 = gallery.allocate_and_insert(conn, p2, image_count=3)
        self.assertEqual(r1, (1, 4))
        self.assertEqual(r2, (4, 7))

    def test_allocation_survives_deleting_the_highest_numbered_row(self):
        # The bug this module was rewritten to avoid: a naive
        # MAX(range_end)-over-live-rows allocator would let a deleted
        # person's range be handed out again. The persisted counter
        # must NOT roll back just because the row that consumed it is
        # gone — those c<N>.jpg files may still exist in MinIO.
        conn = _make_db()
        p1 = gallery.PersonRecord("A", "A", "S", "0", "1")
        p2 = gallery.PersonRecord("B", "B", "S", "0", "2")
        gallery.allocate_and_insert(conn, p1, image_count=3)  # (1, 4)
        gallery.allocate_and_insert(conn, p2, image_count=3)  # (4, 7)

        conn.execute("DELETE FROM brieface WHERE personnelid = '2'")
        conn.commit()

        start, end = gallery.next_free_range(conn, count=3)
        self.assertEqual((start, end), (7, 10), "counter must not roll back after a delete")


class ImageFilenamesTests(unittest.TestCase):
    def test_matches_recognition_engines_block_assumption(self):
        # recognition_engine.py::_decide_person_and_confidence computes
        # pid = (image_num - 1) // 3 + 1 — i.e. it assumes a FIXED,
        # equal-sized block per person. Confirm image_filenames()
        # produces exactly that shape for the default block size.
        filenames = gallery.image_filenames(4, 7)
        self.assertEqual(filenames, ["c4.jpg", "c5.jpg", "c6.jpg"])
        for n in (4, 5, 6):
            pid = (n - 1) // 3 + 1
            self.assertEqual(pid, 2)  # all three belong to the same person block


if __name__ == "__main__":
    unittest.main(verbosity=2)
