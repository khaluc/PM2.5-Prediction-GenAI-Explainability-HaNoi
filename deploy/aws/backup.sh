#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ENV_FILE="${PRODUCTION_ENV_FILE:-.env.production}"
RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-14}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE" >&2
  exit 1
fi

mkdir -p backups
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
temporary="backups/environment-${timestamp}.dump.tmp"
target="backups/environment-${timestamp}.dump"

export PRODUCTION_ENV_FILE="$ENV_FILE"
compose=(docker compose --env-file "$ENV_FILE" -f docker-compose.aws.yml)

"${compose[@]}" exec -T db sh -c \
  'pg_dump --username="$POSTGRES_USER" --dbname="$POSTGRES_DB" --format=custom --no-owner --no-acl' \
  > "$temporary"

if [[ ! -s "$temporary" ]]; then
  rm -f "$temporary"
  echo "Database backup is empty." >&2
  exit 1
fi

mv "$temporary" "$target"
find backups -maxdepth 1 -type f -name 'environment-*.dump' -mtime "+$RETENTION_DAYS" -delete
echo "$target"
