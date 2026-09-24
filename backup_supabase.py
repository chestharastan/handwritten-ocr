#!/usr/bin/env python3
"""Back up a Supabase Postgres database to the local `database/` folder.

Usage:
    python3 backup_supabase.py            # full backup (schema + data)
    python3 backup_supabase.py --csv      # force CSV export of every table

Settings are read from environment variables or a `.env` file next to this
script:
    DATABASE_URL=postgresql://...          (required)
    SUPABASE_URL=https://<ref>.supabase.co (for Storage files)
    SUPABASE_SERVICE_ROLE_KEY=...          (for Storage files, private buckets)

Strategy:
  1. If `pg_dump` is installed, create a full SQL dump (restorable with psql).
  2. Otherwise, fall back to exporting each table to CSV using psycopg.
  3. Download every Storage file into database/storage/<bucket>/<path>.
     Files already downloaded with the same size are skipped, so only new
     files are fetched on later runs.
Table backups are written to database/<timestamp>/ and old ones beyond
KEEP_LAST are removed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKUP_DIR = ROOT / "database"
KEEP_LAST = 10  # number of backups to keep; set to 0 to keep all
SCHEMAS = ["public"]  # schemas to back up in CSV mode


STORAGE_DIR = BACKUP_DIR / "storage"


def load_setting(name: str) -> str | None:
    value = os.environ.get(name)
    env_file = ROOT / ".env"
    if not value and env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith(name + "="):
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
    return value or None


def load_database_url() -> str:
    url = load_setting("DATABASE_URL")
    if not url:
        sys.exit("DATABASE_URL is not set (env var or .env file).")
    return url


def backup_with_pg_dump(url: str, out_dir: Path) -> bool:
    pg_dump = shutil.which("pg_dump")
    if not pg_dump:
        return False
    out_file = out_dir / "backup.sql"
    print(f"Running pg_dump -> {out_file}")
    result = subprocess.run(
        [pg_dump, "--no-owner", "--no-privileges", "--schema=public",
         "--file", str(out_file), url],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("pg_dump failed:\n" + result.stderr.strip())
        out_file.unlink(missing_ok=True)
        return False
    return True


def backup_with_csv(url: str, out_dir: Path) -> None:
    try:
        import psycopg
    except ImportError:
        sys.exit('psycopg is not installed. Run: pip3 install "psycopg[binary]"')

    with psycopg.connect(url) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT table_schema, table_name FROM information_schema.tables
            WHERE table_type = 'BASE TABLE' AND table_schema = ANY(%s)
            ORDER BY table_schema, table_name
            """,
            (SCHEMAS,),
        )
        tables = cur.fetchall()
        if not tables:
            print(f"No tables found in schemas {SCHEMAS}.")
            return

        for schema, table in tables:
            out_file = out_dir / f"{schema}.{table}.csv"
            query = f'COPY "{schema}"."{table}" TO STDOUT WITH (FORMAT csv, HEADER true)'
            rows = 0
            with open(out_file, "wb") as f, cur.copy(query) as copy:
                for chunk in copy:
                    f.write(chunk)
            with open(out_file, "rb") as f:
                rows = max(sum(1 for _ in f) - 1, 0)
            print(f"  {schema}.{table}: ~{rows} rows -> {out_file.name}")


def backup_storage(url: str) -> None:
    api_url = load_setting("SUPABASE_URL")
    key = load_setting("SUPABASE_SERVICE_ROLE_KEY")
    if not api_url or not key:
        print("Skipping Storage files: set SUPABASE_URL and "
              "SUPABASE_SERVICE_ROLE_KEY in .env to back them up.")
        return
    import psycopg

    with psycopg.connect(url) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT bucket_id, name, (metadata->>'size')::bigint FROM storage.objects "
            "WHERE name NOT LIKE '%.emptyFolderPlaceholder' ORDER BY bucket_id, name"
        )
        objects = cur.fetchall()

    print(f"Backing up {len(objects)} Storage files -> {STORAGE_DIR}")
    downloaded = skipped = failed = 0
    for bucket, name, size in objects:
        dest = STORAGE_DIR / bucket / name
        if dest.exists() and (size is None or dest.stat().st_size == size):
            skipped += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        path = urllib.parse.quote(f"{bucket}/{name}")
        req = urllib.request.Request(
            f"{api_url.rstrip('/')}/storage/v1/object/authenticated/{path}",
            headers={"Authorization": f"Bearer {key}", "apikey": key},
        )
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as f:
                shutil.copyfileobj(resp, f)
            tmp.replace(dest)
            downloaded += 1
            print(f"  [{downloaded + skipped}/{len(objects)}] {bucket}/{name}")
        except Exception as e:
            tmp.unlink(missing_ok=True)
            failed += 1
            print(f"  FAILED {bucket}/{name}: {e}")
    print(f"Storage: {downloaded} downloaded, {skipped} already up to date, {failed} failed")


def prune_old_backups() -> None:
    if KEEP_LAST <= 0:
        return
    backups = sorted(p for p in BACKUP_DIR.iterdir() if p.is_dir() and p != STORAGE_DIR)
    for old in backups[:-KEEP_LAST]:
        shutil.rmtree(old)
        print(f"Removed old backup {old.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", action="store_true", help="export tables as CSV")
    parser.add_argument("--no-storage", action="store_true", help="skip Storage files")
    args = parser.parse_args()

    url = load_database_url()
    out_dir = BACKUP_DIR / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.csv or not backup_with_pg_dump(url, out_dir):
        print("Exporting tables to CSV...")
        backup_with_csv(url, out_dir)

    if not args.no_storage:
        backup_storage(url)

    prune_old_backups()
    print(f"Backup complete: {out_dir}")


if __name__ == "__main__":
    main()
