"""Prometheus metrics for batch monitoring and inference instrumentation."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

from prometheus_client import Counter, Gauge, Histogram
from prometheus_client.registry import CollectorRegistry

REQUESTS = "pet_breed_inference_requests_total"
ERRORS = "pet_breed_inference_errors_total"
LATENCY = "pet_breed_inference_latency_seconds"
CONFIDENCE = "pet_breed_inference_confidence"
ABSTENTIONS = "pet_breed_inference_abstentions_total"
DRIFT = "pet_breed_drift_score"
DRIFT_FLAG = "pet_breed_drift_flag"


def create_metrics(registry: CollectorRegistry | None = None) -> dict[str, object]:
    """Create an isolated metric set, useful for exporters and tests."""
    registry = registry or CollectorRegistry()
    return {
        "requests": Counter(
            REQUESTS,
            "Total inference requests.",
            registry=registry,
        ),
        "errors": Counter(
            ERRORS,
            "Total inference errors.",
            registry=registry,
        ),
        "latency": Histogram(
            LATENCY,
            "Inference latency in seconds.",
            registry=registry,
        ),
        "confidence": Gauge(
            CONFIDENCE,
            "Latest observed top prediction confidence.",
            registry=registry,
        ),
        "abstentions": Counter(
            ABSTENTIONS,
            "Total inference abstentions.",
            registry=registry,
        ),
        "drift": Gauge(
            DRIFT,
            "Latest monitoring drift score by detector.",
            ["detector"],
            registry=registry,
        ),
        "drift_flag": Gauge(
            DRIFT_FLAG,
            "Latest monitoring drift flag by detector.",
            ["detector"],
            registry=registry,
        ),
    }


@contextmanager
def observe_inference(metrics: dict[str, object]) -> Iterator[None]:
    """Record request count, latency, and errors around an inference call."""
    requests = metrics["requests"]
    errors = metrics["errors"]
    latency = metrics["latency"]
    assert isinstance(requests, Counter)
    assert isinstance(errors, Counter)
    assert isinstance(latency, Histogram)
    requests.inc()
    started = time.perf_counter()
    try:
        yield
    except Exception:
        errors.inc()
        raise
    finally:
        latency.observe(time.perf_counter() - started)


def record_prediction(
    metrics: dict[str, object],
    confidence: float,
    abstained: bool,
) -> None:
    """Record prediction confidence and abstention state."""
    confidence_metric = metrics["confidence"]
    abstentions = metrics["abstentions"]
    assert isinstance(confidence_metric, Gauge)
    assert isinstance(abstentions, Counter)
    confidence_metric.set(confidence)
    if abstained:
        abstentions.inc()


def set_drift_metrics(
    metrics: dict[str, object],
    detector_scores: dict[str, tuple[float, bool]],
) -> None:
    """Export detector scores and boolean drift flags."""
    drift = metrics["drift"]
    drift_flag = metrics["drift_flag"]
    assert isinstance(drift, Gauge)
    assert isinstance(drift_flag, Gauge)
    for detector, (score, flagged) in detector_scores.items():
        drift.labels(detector=detector).set(score)
        drift_flag.labels(detector=detector).set(int(flagged))


def metric_names() -> set[str]:
    """Return the public metric names used by this module."""
    return {
        REQUESTS,
        ERRORS,
        LATENCY,
        CONFIDENCE,
        ABSTENTIONS,
        DRIFT,
        DRIFT_FLAG,
    }
