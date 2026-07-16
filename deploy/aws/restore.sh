#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 || ! -s "$1" ]]; then
  echo "Usage: bash deploy/aws/restore.sh backups/environment-TIMESTAMP.dump" >&2
  exit 1
fi

if [[ "${CONFIRM_RESTORE:-}" != "yes" ]]; then
  echo "Restore replaces database contents. Re-run with CONFIRM_RESTORE=yes after taking a backup." >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ENV_FILE="${PRODUCTION_ENV_FILE:-.env.production}"
export PRODUCTION_ENV_FILE="$ENV_FILE"
compose=(docker compose --env-file "$ENV_FILE" -f docker-compose.aws.yml)

cat "$1" | "${compose[@]}" exec -T db sh -c \
  'pg_restore --username="$POSTGRES_USER" --dbname="$POSTGRES_DB" --clean --if-exists --no-owner --no-acl'

echo "Restore completed from $1"
