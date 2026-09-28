"""Expose actual drift scorecard values as a Prometheus HTTP exporter."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread

from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from prometheus_client.registry import CollectorRegistry

from pet_breed_mlops.monitoring_metrics import create_metrics, set_drift_metrics


class MetricsHandler(BaseHTTPRequestHandler):
    registry: CollectorRegistry

    def do_GET(self) -> None:
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            return
        payload = generate_latest(self.registry)
        self.send_response(200)
        self.send_header("Content-Type", CONTENT_TYPE_LATEST)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve monitoring metrics for Prometheus.")
    parser.add_argument(
        "--scorecard",
        type=Path,
        default=Path("reports/monitoring/drift_scorecard.json"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scorecard = json.loads(args.scorecard.read_text(encoding="utf-8"))
    metrics = create_metrics()
    detector_scores: dict[str, tuple[float, bool]] = {}

    groups = scorecard.get("groups", [])
    for row in groups:
        key = f'{row["corruption"]}_severity_{row["severity"]}'
        detector_scores[f"{key}_embedding"] = (
            float(row["embedding_mmd"]),
            bool(row["embedding_drift"]),
        )
        detector_scores[f"{key}_pixel"] = (
            float(row["pixel_mmd"]),
            bool(row["pixel_drift"]),
        )
        detector_scores[f"{key}_confidence"] = (
            float(row["confidence_drift"]),
            bool(row["confidence_drift_flag"]),
        )

    set_drift_metrics(metrics, detector_scores)
    MetricsHandler.registry = metrics["requests"]._registry
    server = HTTPServer((args.host, args.port), MetricsHandler)
    print(f"Prometheus metrics: http://{args.host}:{args.port}/metrics")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
