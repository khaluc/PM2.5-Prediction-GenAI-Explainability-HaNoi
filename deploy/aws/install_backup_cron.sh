#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCHEDULE="${BACKUP_CRON_SCHEDULE:-15 2 * * *}"
MARKER="# environment-ai-postgresql-backup"
LOG_FILE="$ROOT_DIR/artifacts/logs/backup.log"
CRON_LINE="$SCHEDULE cd $ROOT_DIR && bash deploy/aws/backup.sh >> $LOG_FILE 2>&1 $MARKER"

if ! command -v crontab >/dev/null 2>&1; then
  echo "crontab is not installed." >&2
  exit 1
fi

mkdir -p "$ROOT_DIR/artifacts/logs"

existing="$(crontab -l 2>/dev/null || true)"
{
  printf '%s\n' "$existing" | grep -Fv "$MARKER" || true
  printf '%s\n' "$CRON_LINE"
} | sed '/^[[:space:]]*$/d' | crontab -

echo "Installed backup schedule:"
crontab -l | grep -F "$MARKER"
