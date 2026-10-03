"""Local webhook receiver used to test the monitoring-to-retraining alert path."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


class RetrainingHandler(BaseHTTPRequestHandler):
    output: Path

    def do_POST(self) -> None:
        if self.path != "/retrain":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        payload = json.loads(body or b"{}")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        self.send_response(202)
        self.end_headers()
        self.wfile.write(b"retraining signal accepted")

    def log_message(self, format: str, *args: object) -> None:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Receive retraining alerts locally.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/monitoring/retraining_alert.json")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RetrainingHandler.output = args.output
    server = HTTPServer((args.host, args.port), RetrainingHandler)
    print(f"Retraining webhook: http://{args.host}:{args.port}/retrain")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
