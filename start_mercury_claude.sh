#!/bin/bash
# Start Claude Code with Mercury 2 backend
# Usage: ./start_mercury_claude.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROXY_PORT=8082
PROXY_PID=""

# Mercury 2 API key
export INCEPTION_API_KEY="${INCEPTION_API_KEY:-sk_7a760ef3d2444f756cb947511d21b9d5}"

cleanup() {
    if [ -n "$PROXY_PID" ] && kill -0 "$PROXY_PID" 2>/dev/null; then
        echo "Stopping Mercury proxy (PID $PROXY_PID)..."
        kill "$PROXY_PID" 2>/dev/null
        wait "$PROXY_PID" 2>/dev/null
    fi
}
trap cleanup EXIT

echo "=== Starting Mercury 2 Proxy ==="
echo "Model: mercury-2 | Reasoning: high"
echo ""

# Start proxy in background
python3 "$SCRIPT_DIR/mercury_proxy.py" --port "$PROXY_PORT" &
PROXY_PID=$!

# Wait for proxy to be ready
sleep 1
if ! kill -0 "$PROXY_PID" 2>/dev/null; then
    echo "ERROR: Proxy failed to start"
    exit 1
fi

echo ""
echo "=== Proxy running on http://127.0.0.1:$PROXY_PORT ==="
echo ""

# Launch Claude Code with Mercury backend
export ANTHROPIC_BASE_URL="http://127.0.0.1:$PROXY_PORT"
export ANTHROPIC_API_KEY="mercury-proxy"

echo "Starting Claude Code with Mercury 2 backend..."
echo ""
claude "$@"
