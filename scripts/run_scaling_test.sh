#!/usr/bin/env bash
# Run the same Locust flood against 1, 2 and 3 API containers and collect CSVs.
# Usage: bash scripts/run_scaling_test.sh [users] [spawn_rate] [duration]
set -euo pipefail
USERS=${1:-100}; RATE=${2:-10}; DURATION=${3:-60s}
mkdir -p results

for N in 1 2 3; do
  echo "=== $N API container(s) ==="
  docker compose up -d --scale api="$N" --build
  sleep 20                                   # let TensorFlow load the model
  locust -f locust/locustfile.py --host http://localhost:8080 \
         --headless -u "$USERS" -r "$RATE" -t "$DURATION" --csv "results/containers_$N" --only-summary
done
docker compose up -d --scale api=1
python scripts/summarize_results.py
