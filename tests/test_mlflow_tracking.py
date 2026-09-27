from pathlib import Path

import mlflow

from pet_breed_mlops.tracking.mlflow_tracker import MLflowTracker


def test_mlflow_tracker_logs_run(tmp_path: Path) -> None:
    database_path = (tmp_path / "mlflow.db").as_posix()
    tracking_uri = f"sqlite:///{database_path}"
    tracker = MLflowTracker(
        tracking_uri=tracking_uri,
        experiment_name="test-pet-breed",
        registered_model_name="TestPetBreedClassifier",
    )
    tracker.start(run_name="test-run")
    tracker.log_params({"backbone": "resnet18", "batch_size": 32})
    tracker.log_metrics({"val_top_1_accuracy": 0.5})
    run_id = tracker.run_id
    tracker.finish()

    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)
    assert run.data.params["backbone"] == "resnet18"
    assert run.data.metrics["val_top_1_accuracy"] == 0.5
