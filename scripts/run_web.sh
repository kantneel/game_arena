#!/bin/bash
# Run Game Arena Web Dashboard (backend + frontend)
#
# Usage: ./scripts/run_web.sh
#
# Prerequisites:
#   - Python venv with game_arena[web] installed
#   - Bun installed for frontend
#   - Run from the game_arena root directory

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🏟️  Game Arena Web Dashboard${NC}"
echo ""

# Check if bun is installed
if ! command -v bun &> /dev/null; then
    echo -e "${RED}Error: bun is not installed.${NC}"
    echo "Install with: curl -fsSL https://bun.sh/install | bash"
    exit 1
fi

# Check if frontend dependencies are installed
if [ ! -d "web/frontend/node_modules" ]; then
    echo -e "${YELLOW}Installing frontend dependencies...${NC}"
    cd web/frontend
    bun install
    cd "$PROJECT_ROOT"
fi

# Function to cleanup background processes on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down...${NC}"
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start backend
echo -e "${GREEN}Starting backend on http://localhost:8000${NC}"
cd web/backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
cd "$PROJECT_ROOT"

# Wait for backend to start
sleep 2

# Start frontend
echo -e "${GREEN}Starting frontend on http://localhost:3000${NC}"
cd web/frontend
bun dev &
FRONTEND_PID=$!
cd "$PROJECT_ROOT"

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✓ Dashboard running!${NC}"
echo ""
echo -e "  Frontend:  ${BLUE}http://localhost:3000${NC}"
echo -e "  Backend:   ${BLUE}http://localhost:8000${NC}"
echo -e "  API Docs:  ${BLUE}http://localhost:8000/docs${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID

