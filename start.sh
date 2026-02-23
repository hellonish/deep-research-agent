#!/bin/bash
echo "Starting Wort Infrastructure (Redis)..."

# Spin up Redis
docker-compose up -d

echo "Starting Wort Backend and Frontend..."

# Start backend in background
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000 &
BACKEND_PID=$!

# Start frontend
cd wort-frontend
npm run dev &
FRONTEND_PID=$!

# Wait forever or until interrupted
wait

# Trap CTRL+C to kill everything
trap "echo 'Shutting down...'; kill $BACKEND_PID $FRONTEND_PID; docker-compose down" EXIT INT TERM
