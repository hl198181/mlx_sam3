#!/bin/bash

# SAM3 Segmentation Studio - Development Server Launcher
# Starts both backend (FastAPI) and frontend (Next.js) servers

# Don't exit on error in cleanup - we want to clean up even if something fails
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
NC='\033[0m' # No Color

# Parse arguments
USE_GPU=false
for arg in "$@"; do
    case $arg in
        --gpu)
            USE_GPU=true
            shift
            ;;
    esac
done

if [ "$USE_GPU" = true ]; then
    echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  SAM3 Segmentation Studio (GPU/CUDA)   ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
    BACKEND_SCRIPT="main_gpu.py"
    BACKEND_PORT=8000
else
    echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║   SAM3 Segmentation Studio (MLX)       ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
    BACKEND_SCRIPT="main.py"
    BACKEND_PORT=8000
fi
echo ""

# Load environment variables from .env.local if it exists
if [ -f "$SCRIPT_DIR/.env.local" ]; then
    set -a
    source "$SCRIPT_DIR/.env.local"
    set +a
fi

# Array to store process PIDs
PIDS=()

# Function to cleanup background processes on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down servers...${NC}"
    
    # Send SIGTERM to all processes for graceful shutdown
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    
    # Wait up to 5 seconds for graceful shutdown
    sleep 2
    
    # Force kill any remaining processes
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "${YELLOW}Force killing process $pid...${NC}"
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
    
    # Also kill any remaining jobs in this shell (macOS compatible)
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

# Initialize fnm (for node/npm) if available
FNM_PATH="${HOME}/.local/share/fnm"
if [ -d "$FNM_PATH" ]; then
    export PATH="$FNM_PATH:$PATH"
    eval "$(fnm env --shell bash)"
elif [ -f "$HOME/.local/bin/fnm" ]; then
    export PATH="$HOME/.local/bin:$PATH"
    eval "$(fnm env --shell bash)"
fi

# Install backend dependencies using uv (into the project's venv)
echo -e "${YELLOW}Ensuring backend dependencies...${NC}"
cd "$PROJECT_ROOT"
uv pip install -r "$BACKEND_DIR/requirements.txt" --quiet

# Check if frontend dependencies are installed
if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    echo -e "${YELLOW}Installing frontend dependencies...${NC}"
    cd "$FRONTEND_DIR" && npm install
    cd "$SCRIPT_DIR"
fi

echo -e "${GREEN}Starting Backend (FastAPI) on http://localhost:${BACKEND_PORT}${NC}"
cd "$PROJECT_ROOT"

# Set the API URL for frontend to connect to the correct backend
# Use hostname so browser can reach the backend when accessed remotely
HOSTNAME=$(hostname -f 2>/dev/null || hostname)
if command -v ip &>/dev/null; then
    LOCAL_IPS=$(ip -4 addr show 2>/dev/null | awk '/inet / {split($2,a,"/"); print a[1]}' | grep -v '127.0.0.1' | tr '\n' ',' | sed 's/,$//')
else
    LOCAL_IPS=$(ifconfig 2>/dev/null | awk '/inet / && !/127.0.0.1/ {print $2}' | tr '\n' ',' | sed 's/,$//')
fi
export NEXT_PUBLIC_API_URL="http://${HOSTNAME}:${BACKEND_PORT}"
export ALLOWED_DEV_ORIGINS="${HOSTNAME},${LOCAL_IPS}"

uv run python "$BACKEND_DIR/$BACKEND_SCRIPT" &
BACKEND_PID=$!
PIDS+=($BACKEND_PID)

# Wait a moment for backend to start
sleep 2

echo -e "${GREEN}Starting Frontend (Next.js) on http://localhost:3000${NC}"
cd "$FRONTEND_DIR"
npm run dev &
FRONTEND_PID=$!
PIDS+=($FRONTEND_PID)

echo ""
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo -e "${GREEN}  Servers are running!${NC}"
echo -e "${GREEN}  Frontend: http://localhost:3000${NC}"
echo -e "${GREEN}  Backend:  http://localhost:${BACKEND_PORT}${NC}"
echo -e "${GREEN}  API Docs: http://localhost:${BACKEND_PORT}/docs${NC}"
if [ "$USE_GPU" = true ]; then
echo -e "${GREEN}  Mode:     GPU (PyTorch/CUDA)${NC}"
else
echo -e "${GREEN}  Mode:     MLX (Apple Silicon)${NC}"
fi
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all servers${NC}"
echo ""

# Wait for all processes (will exit when any process exits or on Ctrl+C)
# The EXIT trap will ensure cleanup happens
set +e  # Temporarily disable exit on error for wait
wait "${PIDS[@]}" 2>/dev/null
set -e  # Re-enable exit on error

