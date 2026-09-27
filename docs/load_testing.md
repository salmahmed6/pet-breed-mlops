# Inference Load Testing and Deployment Evidence

## Batch scoring
Run:
    python -m scripts.batch_inference --checkpoint artifacts/models/resnet18_best.pt --batch-size 32 --max-batches 50 --output reports/serving/batch_inference.json

The report records actual device, images, elapsed time, images/second, and validation top-1. No measurements are committed before execution.

## Locust
Install:
    pip install locust

Start the serving endpoint first. Windows:
    .\scripts\run_load_test.ps1 -HostUrl http://127.0.0.1:3000 -Users 10 -DurationSeconds 60 -Payload tests/fixtures/pet.jpg

Linux/macOS:
    HOST_URL=http://127.0.0.1:3000 USERS=10 DURATION_SECONDS=60 PET_BREED_PAYLOAD=tests/fixtures/pet.jpg ./scripts/run_load_test.sh

Locust captures requests, failures, requests/second, and latency percentiles under reports/serving/locust*. The config records concurrency, duration, payload, endpoint, and batch size. Unavailable endpoints or missing payloads fail the test; no results are fabricated.

## Deployment evidence
Issue #32 supplies the serving contract. This issue supplies batch scoring, reproducible load generation, and evidence collection.
