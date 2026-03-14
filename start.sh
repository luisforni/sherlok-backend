#!/usr/bin/env bash
set -e
echo "Starting voice-conscience backend..."
export $(grep -v '^#' .env.example | xargs)
uvicorn main:app --host ${BACKEND_HOST:-0.0.0.0} --port ${BACKEND_PORT:-8000}
