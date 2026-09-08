#!/usr/bin/env bash
# Interactive: point the Fact Knowledge Layer at Groq's free API.
#
#   ./scripts/setup-groq.sh
#
# Writes ./.env (git-ignored). Re-run any time to change the key. Nothing here
# is required - with no .env the system runs entirely on the local model.

set -euo pipefail
cd "$(dirname "$0")/.."
ENV_FILE=".env"

cat <<'INTRO'
--------------------------------------------------------------------------------
 Groq free API  -  no credit card, ~100x faster than local CPU inference.

  1. Open   https://console.groq.com   and sign in (Google or GitHub).
  2. Left sidebar  ->  "API Keys"  ->  "Create API Key".
  3. Copy the key (it starts with  gsk_  and is shown only once).
--------------------------------------------------------------------------------
INTRO

read -rp "Paste your Groq API key (or leave blank to cancel): " KEY
if [ -z "${KEY:-}" ]; then
  echo "cancelled - no changes made."
  exit 0
fi
case "$KEY" in
  gsk_*) : ;;
  *) echo "warning: a Groq key normally starts with 'gsk_'. Writing it anyway." ;;
esac

read -rp "Model [llama-3.3-70b-versatile]: " MODEL
MODEL="${MODEL:-llama-3.3-70b-versatile}"

# Preserve any non-Groq lines already in .env
if [ -f "$ENV_FILE" ]; then
  grep -vE '^(LLM_BACKEND|GROQ_API_KEY|FKL_GROQ_MODEL)=' "$ENV_FILE" > "$ENV_FILE.tmp" || true
  mv "$ENV_FILE.tmp" "$ENV_FILE"
fi
{
  echo "LLM_BACKEND=groq"
  echo "GROQ_API_KEY=$KEY"
  echo "FKL_GROQ_MODEL=$MODEL"
} >> "$ENV_FILE"

echo
echo "wrote $ENV_FILE"
echo "verify:  python run.py status      (backend line should read 'groq')"
echo "run:     python run.py extract && python run.py link"
