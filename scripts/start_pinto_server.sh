#!/bin/bash
# Pinto's real MCP file tools on a local in-memory database (no network, no Supabase).
# Leave it running while run.py uses the pinto condition. Port defaults to 8791.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/runs/mcp-logs"
cd "$ROOT/pinto-notes/supabase/functions/mcp"
PINTO_REPO="$ROOT/pinto-notes" PINTO_LOG_DIR="$ROOT/runs/mcp-logs" exec deno run -A "$ROOT/scripts/pinto_mcp_local.ts" "${1:-8791}"
