"""Append-only decision store with a hash chain, backed by SQLite.

Each record stores the hash of the previous record, so editing or deleting a
record breaks the chain and verify_chain() reports it. Writes are idempotent by
case id. ASCII only.
"""
import hashlib
import json
import sqlite3

_GENESIS = "0" * 64


def _canonical(record):
    return json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)


class DecisionStore:
    def __init__(self, path=":memory:"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS decisions ("
            "seq INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT UNIQUE NOT NULL, "
            "record TEXT NOT NULL, prev_hash TEXT NOT NULL, hash TEXT NOT NULL)")
        self.conn.commit()

    def _last_hash(self):
        row = self.conn.execute(
            "SELECT hash FROM decisions ORDER BY seq DESC LIMIT 1").fetchone()
        return row[0] if row else _GENESIS

    def append(self, case_id, record):
        """Insert a record. Returns False if the case id already exists."""
        if self.get(case_id) is not None:
            return False
        prev = self._last_hash()
        body = _canonical(record)
        digest = hashlib.sha256((prev + body).encode("utf-8")).hexdigest()
        self.conn.execute(
            "INSERT INTO decisions (case_id, record, prev_hash, hash) VALUES (?,?,?,?)",
            (case_id, body, prev, digest))
        self.conn.commit()
        return True

    def get(self, case_id):
        row = self.conn.execute(
            "SELECT record FROM decisions WHERE case_id = ?", (case_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def count(self):
        return self.conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]

    def all_records(self):
        rows = self.conn.execute("SELECT record FROM decisions ORDER BY seq").fetchall()
        return [json.loads(r[0]) for r in rows]

    def verify_chain(self):
        """True if every record hashes correctly and links to the one before it."""
        prev = _GENESIS
        for record, prev_hash, digest in self.conn.execute(
                "SELECT record, prev_hash, hash FROM decisions ORDER BY seq"):
            expected = hashlib.sha256((prev + record).encode("utf-8")).hexdigest()
            if prev_hash != prev or digest != expected:
                return False
            prev = digest
        return True
