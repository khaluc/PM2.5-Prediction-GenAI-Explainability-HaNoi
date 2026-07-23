#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 2 || ! -s "$1" ]]; then
  echo "Usage: prepare_environment.sh SOURCE_SECRETS TARGET_ENV" >&2
  exit 1
fi

SOURCE_SECRETS="$1"
TARGET_ENV="$2"
ROOT_DIR="$(cd "$(dirname "$TARGET_ENV")" && pwd)"
EXAMPLE_ENV="$ROOT_DIR/.env.production.example"

if [[ ! -f "$EXAMPLE_ENV" ]]; then
  echo "Missing $EXAMPLE_ENV" >&2
  exit 1
fi

cp "$EXAMPLE_ENV" "$TARGET_ENV"

allowed_keys=(
  TOMTOM_API_KEY
  DASHSCOPE_API_KEY
  DASHSCOPE_BASE_URL
  DASHSCOPE_MODEL
  DASHSCOPE_TIMEOUT_SECONDS
  DASHSCOPE_MAX_OUTPUT_TOKENS
  DASHSCOPE_TEMPERATURE
  DASHSCOPE_THINKING_ENABLED
)

for key in "${allowed_keys[@]}"; do
  line="$(grep -m1 "^${key}=" "$SOURCE_SECRETS" || true)"
  if [[ -n "$line" ]]; then
    sed -i "/^${key}=/d" "$TARGET_ENV"
    printf '%s\n' "$line" >> "$TARGET_ENV"
  fi
done

postgres_password="$(openssl rand -hex 32)"
sed -i "s/^POSTGRES_PASSWORD=.*/POSTGRES_PASSWORD=${postgres_password}/" "$TARGET_ENV"
chmod 600 "$TARGET_ENV"

for required_key in POSTGRES_PASSWORD TOMTOM_API_KEY DASHSCOPE_API_KEY; do
  if ! grep -q "^${required_key}=." "$TARGET_ENV"; then
    echo "Missing required value: $required_key" >&2
    exit 1
  fi
done

rm -f "$SOURCE_SECRETS"
echo "Production environment is ready."
