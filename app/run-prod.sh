#!/bin/bash

# SAM3 Segmentation Studio - Production Server
# Starts backend (FastAPI) and frontend (Next.js production build)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Load environment variables from .env.local
if [ -f "$SCRIPT_DIR/.env.local" ]; then
    set -a
    source "$SCRIPT_DIR/.env.local"
    set +a
fi

# Array to store process PIDs
PIDS=()

cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down servers...${NC}"
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    sleep 2
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
    local remaining_jobs
    remaining_jobs=$(jobs -p 2>/dev/null)
    if [ -n "$remaining_jobs" ]; then
        echo "$remaining_jobs" | xargs kill -TERM 2>/dev/null || true
        sleep 1
        remaining_jobs=$(jobs -p 2>/dev/null)
        if [ -n "$remaining_jobs" ]; then
            echo "$remaining_jobs" | xargs kill -KILL 2>/dev/null || true
        fi
    fi
    echo -e "${GREEN}All servers stopped.${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  SAM3 Segmentation Studio (Production) ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# Build frontend
echo -e "${YELLOW}Building frontend...${NC}"
cd "$FRONTEND_DIR"
npm run build

# Start backend
echo -e "${GREEN}Starting Backend (FastAPI) on http://0.0.0.0:8000${NC}"
cd "$PROJECT_ROOT"
uv run python "$BACKEND_DIR/main.py" &
BACKEND_PID=$!
PIDS+=($BACKEND_PID)

# Wait for backend to be ready
echo -e "${YELLOW}Waiting for backend to be ready...${NC}"
for i in $(seq 1 30); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}Backend is ready.${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}Backend failed to start within 30 seconds.${NC}"
        exit 1
    fi
    sleep 1
done

# Start frontend (production mode)
echo -e "${GREEN}Starting Frontend (Next.js) on http://0.0.0.0:3000${NC}"
cd "$FRONTEND_DIR"
npm run start &
FRONTEND_PID=$!
PIDS+=($FRONTEND_PID)

echo ""
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo -e "${GREEN}  Servers are running!${NC}"
echo -e "${GREEN}  Frontend: http://localhost:3000${NC}"
echo -e "${GREEN}  Backend:  http://localhost:8000${NC}"
echo -e "${GREEN}  API Docs: http://localhost:8000/docs${NC}"
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo ""

set +e
wait "${PIDS[@]}" 2>/dev/null
set -e
