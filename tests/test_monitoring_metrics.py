from prometheus_client.registry import CollectorRegistry

from pet_breed_mlops.monitoring_metrics import (
    ERRORS,
    REQUESTS,
    create_metrics,
    metric_names,
    observe_inference,
    record_prediction,
    set_drift_metrics,
)


def test_metric_names_are_stable() -> None:
    names = metric_names()
    assert {REQUESTS, ERRORS}.issubset(names)


def test_metrics_record_inference_and_drift() -> None:
    registry = CollectorRegistry()
    metrics = create_metrics(registry)

    with observe_inference(metrics):
        record_prediction(metrics, confidence=0.8, abstained=True)

    set_drift_metrics(metrics, {"embedding": (0.2, True)})

    assert registry.get_sample_value(REQUESTS) == 1.0
    assert registry.get_sample_value(ERRORS) == 0.0
    assert registry.get_sample_value("pet_breed_inference_abstentions_total") == 1.0
    assert registry.get_sample_value("pet_breed_drift_score", {"detector": "embedding"}) == 0.2
    assert registry.get_sample_value("pet_breed_drift_flag", {"detector": "embedding"}) == 1.0


def test_metrics_record_errors() -> None:
    registry = CollectorRegistry()
    metrics = create_metrics(registry)

    try:
        with observe_inference(metrics):
            raise RuntimeError("expected")
    except RuntimeError:
        pass

    assert registry.get_sample_value(REQUESTS) == 1.0
    assert registry.get_sample_value(ERRORS) == 1.0
