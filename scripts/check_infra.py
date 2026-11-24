"""Checks the local infrastructure:  python scripts/check_infra.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pg8000.dbapi as pg  # noqa: E402  (pure-Python driver, no DLL)
from qdrant_client import QdrantClient  # noqa: E402

from telixia.core.config import get_settings  # noqa: E402

EXPECTED = {"analytics", "catalog", "market", "docs", "audit"}


def check_admin(s):
    try:
        conn = pg.connect(**s.pg_params())
        cur = conn.cursor()
        cur.execute("SHOW server_version")
        version = cur.fetchone()[0]
        cur.execute("SELECT schema_name FROM information_schema.schemata")
        missing = EXPECTED - {r[0] for r in cur.fetchall()}
        conn.close()
        if missing:
            print(f"[FAIL] PostgreSQL {version}: missing schemas {missing}")
            return False
        print(f"[OK] PostgreSQL {version}: 5 schemas present")
        return True
    except Exception as e:
        print(f"[FAIL] PostgreSQL admin: {e}")
        return False


def check_readonly(s):
    try:
        conn = pg.connect(**s.pg_params(read_only=True))
        cur = conn.cursor()
        cur.execute("SELECT 1")
        try:
            cur.execute("CREATE TABLE analytics.should_fail(x int)")
            print("[FAIL] read-only role was able to create a table!")
            conn.close()
            return False
        except pg.DatabaseError:
            conn.rollback()
        conn.close()
        print("[OK] telixia_ro: SELECT allowed, writes refused")
        return True
    except Exception as e:
        print(f"[FAIL] read-only role: {e}")
        return False


def check_qdrant(s):
    try:
        client = QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key.get_secret_value(), timeout=5)
        cols = [c.name for c in client.get_collections().collections]
        print(f"[OK] Qdrant at {s.qdrant_url}: collections {cols}")
        return True
    except Exception as e:
        print(f"[FAIL] Qdrant: {e}")
        return False


if __name__ == "__main__":
    s = get_settings()
    results = [check_admin(s), check_readonly(s), check_qdrant(s)]
    print("\nInfrastructure ready." if all(results) else "\nSome checks failed.")
    sys.exit(0 if all(results) else 1)