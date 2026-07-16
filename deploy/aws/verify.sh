#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ENV_FILE="${PRODUCTION_ENV_FILE:-.env.production}"
export PRODUCTION_ENV_FILE="$ENV_FILE"
compose=(docker compose --env-file "$ENV_FILE" -f docker-compose.aws.yml)

"${compose[@]}" ps

"${compose[@]}" exec -T api python -c \
  "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=15).read().decode())"

"${compose[@]}" exec -T api python -c \
  "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/system/database', timeout=15).read().decode())"

"${compose[@]}" exec -T api python -c \
  "from src.models.predict import load_forecast_artifact; a=load_forecast_artifact(); print({'model': a['model_name'], 'horizons': a['horizons']})"

curl --fail --silent --show-error http://127.0.0.1/nginx-health
