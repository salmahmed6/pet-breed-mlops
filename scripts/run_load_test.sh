#!/usr/bin/env bash
set -euo pipefail
HOST_URL="${HOST_URL:-http://127.0.0.1:3000}"; USERS="${USERS:-10}"; DURATION_SECONDS="${DURATION_SECONDS:-60}"; PAYLOAD="${PET_BREED_PAYLOAD:-tests/fixtures/pet.jpg}"
export PET_BREED_PAYLOAD="$PAYLOAD"
locust -f locustfile.py --headless -H "$HOST_URL" -u "$USERS" -r "$USERS" -t "${DURATION_SECONDS}s" --csv reports/serving/locust --json
