#!/usr/bin/env bash
# Interactive: point the Fact Knowledge Layer at Google's Gemini (AI Studio).
#
#   ./scripts/setup-gemini.sh
#
# Writes ./.env (git-ignored). Nothing here is required - with no .env the
# system runs entirely on the local model. The Gemini free key needs no card.

set -euo pipefail
cd "$(dirname "$0")/.."
ENV_FILE=".env"

cat <<'INTRO'
--------------------------------------------------------------------------------
 Google AI Studio (Gemini)  -  free key, NO credit card.

  1. Open   https://aistudio.google.com/apikey   and sign in (Google account).
  2. Click  "Create API key".
  3. Copy the key.
--------------------------------------------------------------------------------
INTRO

read -rp "Paste your Gemini API key (or leave blank to cancel): " KEY
if [ -z "${KEY:-}" ]; then
  echo "cancelled - no changes made."
  exit 0
fi

read -rp "Model [gemini-2.0-flash]: " MODEL
MODEL="${MODEL:-gemini-2.0-flash}"

# Drop any earlier backend lines so exactly one backend is active.
if [ -f "$ENV_FILE" ]; then
  grep -vE '^(LLM_BACKEND|GROQ_API_KEY|FKL_GROQ_MODEL|GEMINI_API_KEY|GOOGLE_API_KEY|FKL_GEMINI_MODEL)=' \
    "$ENV_FILE" > "$ENV_FILE.tmp" || true
  mv "$ENV_FILE.tmp" "$ENV_FILE"
fi
{
  echo "LLM_BACKEND=gemini"
  echo "GEMINI_API_KEY=$KEY"
  echo "FKL_GEMINI_MODEL=$MODEL"
} >> "$ENV_FILE"

echo
echo "wrote $ENV_FILE"
echo "verify:  python run.py status      (backend line should read 'gemini')"
echo "run:     python run.py extract && python run.py link"
