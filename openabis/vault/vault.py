"""Zero-Trust Biometric Vault: UUID-first storage, PII decoupling.

White paper §5 — the vector tier holds ONLY {uuid -> 512-D vector}.
Names, credentials, passport numbers live in this isolated relational
layer (firewalled + AES-256 at rest in production; this facade uses local
SQLite so the pattern is runnable anywhere).
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List



@dataclass
class VaultEntry:
    vector_uuid: str
    pii_refs: List[str]              # hashed references — no raw PII at rest
    metadata: Dict[str, Any]


class VectorVault:
    """Reference impl of the PII-decoupling pattern.

    * Only `/vector_uuid -> {hash(pii_field), weak metadata}` are written.
    * `reveal_identity()` is the only PII-resolving call and is gated behind
      an explicit audit log in production deployments.
    """

    def __init__(self, db_path: str = "vault.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._migrate()

    def _migrate(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS vault_entries (
                vector_uuid TEXT PRIMARY KEY,
                pii_refs     TEXT,
                metadata     TEXT,
                created_at   TEXT DEFAULT (datetime('now'))
            )
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------ #
    @staticmethod
    def hash_ref(raw: str) -> str:
        """One-way PII reference — reversible only via the isolated raw store."""
        salt = secrets.token_bytes(4)
        return salt.hex() + hashlib.sha256(salt + raw.encode("utf-8")).hexdigest()[:40]

    def link(self, vector_uuid: str, pii: Dict[str, str], metadata: Dict[str, Any] | None = None) -> bool:
        """Associate a vector UUID with hashed PII references."""
        cur = self._conn.cursor()
        refs = [self.hash_ref(f"{k}={v}") for k, v in sorted(pii.items())]
        cur.execute(
            """INSERT INTO vault_entries (vector_uuid, pii_refs, metadata)
                   VALUES (?, ?, ?)
                   ON CONFLICT(vector_uuid) DO UPDATE SET
                       pii_refs = excluded.pii_refs,
                       metadata = excluded.metadata
            """,
            (vector_uuid, json.dumps(refs), json.dumps(metadata or {})),
        )
        self._conn.commit()
        return True

    def unlink(self, vector_uuid: str) -> int:
        """GDPR / BIPA erasure hook — drop the PII↔UUID relation."""
        cur = self._conn.cursor()
        cur.execute("DELETE FROM vault_entries WHERE vector_uuid = ?", (vector_uuid,))
        self._conn.commit()
        return cur.rowcount

    def lookup(self, vector_uuid: str) -> VaultEntry | None:
        cur = self._conn.cursor()
        cur.execute("SELECT vector_uuid, pii_refs, metadata FROM vault_entries WHERE vector_uuid = ?", (vector_uuid,))
        row = cur.fetchone()
        if row is None:
            return None
        return VaultEntry(
            vector_uuid=row[0],
            pii_refs=json.loads(row[1]),
            metadata=json.loads(row[2]) if row[2] else {},
        )

    def count(self) -> int:
        cur = self._conn.cursor()
        cur.execute("SELECT COUNT(*) FROM vault_entries")
        return int(cur.fetchone()[0])

    def close(self) -> None:
        self._conn.close()
