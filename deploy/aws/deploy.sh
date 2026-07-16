#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ENV_FILE="${PRODUCTION_ENV_FILE:-.env.production}"
COMPOSE_FILE="docker-compose.aws.yml"

if [[ ! -f "$ENV_FILE" ]]; then
  cp .env.production.example "$ENV_FILE"
  echo "Created $ENV_FILE. Fill its secrets, then run this script again." >&2
  exit 1
fi

for command_name in docker curl; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Missing required command: $command_name" >&2
    exit 1
  fi
done

if ! docker compose version >/dev/null 2>&1; then
  echo "Docker Compose plugin is required." >&2
  exit 1
fi

read_env() {
  local key="$1"
  sed -n "s/^${key}=//p" "$ENV_FILE" | tail -n 1 | tr -d '\r'
}

postgres_password="$(read_env POSTGRES_PASSWORD)"
if [[ -z "$postgres_password" || "$postgres_password" == CHANGE_TO_* ]]; then
  echo "Set a strong POSTGRES_PASSWORD in $ENV_FILE." >&2
  exit 1
fi

mkdir -p backups artifacts/logs artifacts/alerts artifacts/reports output/pdf

export PRODUCTION_ENV_FILE="$ENV_FILE"
compose=(docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE")

"${compose[@]}" config --quiet
"${compose[@]}" build
"${compose[@]}" run --no-deps --rm --user 0:0 api \
  sh -c "chown -R app:app /app/data /app/artifacts /app/output"
"${compose[@]}" up -d --remove-orphans

for attempt in $(seq 1 60); do
  if curl --fail --silent --show-error http://127.0.0.1/nginx-health >/dev/null; then
    echo "Environment AI is healthy."
    "${compose[@]}" ps
    exit 0
  fi
  sleep 5
done

echo "Deployment did not become healthy in time." >&2
"${compose[@]}" ps >&2
"${compose[@]}" logs --tail=100 api dashboard nginx >&2
exit 1
