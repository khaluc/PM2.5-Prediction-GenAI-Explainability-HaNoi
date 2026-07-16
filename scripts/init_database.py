"""Migrate PostgreSQL and idempotently import existing CSV datasets."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from alembic import command
from alembic.config import Config
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.connection import get_database_url, ping_database  # noqa: E402
from src.database.writer import DatabaseWriter  # noqa: E402


LOGGER = logging.getLogger("database_init")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", help="Override DATABASE_URL")
    parser.add_argument(
        "--history", default="data/processed/air_quality_clean.csv", help="Clean merged history"
    )
    parser.add_argument("--live-air", default="data/raw/air_quality.csv")
    parser.add_argument("--live-weather", default="data/raw/weather.csv")
    parser.add_argument("--traffic", default="data/raw/traffic.csv")
    parser.add_argument("--alerts", default="artifacts/alerts/alerts.json")
    parser.add_argument("--chunk-size", type=int, default=5_000)
    parser.add_argument("--migrate-only", action="store_true")
    parser.add_argument("--force-import", action="store_true")
    return parser.parse_args()


def migrate(database_url: str) -> None:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    command.upgrade(config, "head")


def _import_chunks(
    path: Path,
    chunk_size: int,
    importer,
    label: str,
) -> dict[str, int]:
    totals: dict[str, int] = {}
    if not path.exists():
        LOGGER.warning("Skip missing %s file: %s", label, path)
        return totals
    for index, frame in enumerate(pd.read_csv(path, chunksize=chunk_size, low_memory=False), start=1):
        imported = importer(frame)
        if isinstance(imported, dict):
            for key, value in imported.items():
                totals[key] = totals.get(key, 0) + int(value)
        else:
            totals[label] = totals.get(label, 0) + int(imported)
        if index == 1 or index % 10 == 0:
            LOGGER.info("%s: processed %,d rows", label, sum(totals.values()))
    return totals


def import_existing_data(args: argparse.Namespace, writer: DatabaseWriter) -> dict[str, int]:
    previous = writer.get_state("initial_import")
    alerts_imported = 0
    alerts_path = PROJECT_ROOT / args.alerts
    if alerts_path.exists():
        payload = json.loads(alerts_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            for alert in payload:
                if isinstance(alert, dict) and alert.get("alert_id"):
                    alerts_imported += writer.upsert_alert(alert)
    if previous and previous.get("completed") and not args.force_import:
        LOGGER.info("Initial import already completed; use --force-import to re-run it.")
        rows = {key: int(value) for key, value in (previous.get("rows") or {}).items()}
        rows["alerts"] = alerts_imported
        writer.set_state("initial_import", {**previous, "rows": rows})
        return rows

    writer.set_state(
        "initial_import",
        {"completed": False, "started_at": datetime.now(timezone.utc).isoformat()},
    )
    totals: dict[str, int] = {}
    totals["alerts"] = alerts_imported

    def merge(values: dict[str, int]) -> None:
        for key, value in values.items():
            totals[key] = totals.get(key, 0) + int(value)

    merge(
        _import_chunks(
            PROJECT_ROOT / args.history,
            args.chunk_size,
            writer.upsert_monitoring,
            "history",
        )
    )
    merge(
        _import_chunks(
            PROJECT_ROOT / args.live_air,
            args.chunk_size,
            writer.upsert_air_quality,
            "live_air",
        )
    )
    merge(
        _import_chunks(
            PROJECT_ROOT / args.live_weather,
            args.chunk_size,
            writer.upsert_weather,
            "live_weather",
        )
    )
    merge(
        _import_chunks(
            PROJECT_ROOT / args.traffic,
            args.chunk_size,
            writer.upsert_traffic,
            "traffic",
        )
    )
    writer.set_state(
        "initial_import",
        {
            "completed": True,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "rows": totals,
        },
    )
    return totals


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_dotenv(override=False)
    database_url = args.database_url or get_database_url()
    migrate(database_url)
    ping_database(database_url)
    LOGGER.info("Database migration complete and connection healthy.")
    if args.migrate_only:
        return
    totals = import_existing_data(args, DatabaseWriter(database_url, batch_size=args.chunk_size))
    LOGGER.info("Database import complete: %s", totals)


if __name__ == "__main__":
    main()
