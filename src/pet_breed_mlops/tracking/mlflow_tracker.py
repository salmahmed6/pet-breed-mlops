from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import yaml


class MLflowTracker:
    """Small wrapper around MLflow for reproducible training runs."""

    def __init__(
        self,
        *,
        tracking_uri: str,
        experiment_name: str,
        registered_model_name: str,
    ) -> None:
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.registered_model_name = registered_model_name
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.run = None

    @classmethod
    def from_config(cls, config_path: str | Path) -> "MLflowTracker":
        with Path(config_path).open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        return cls(
            tracking_uri=str(config["tracking_uri"]),
            experiment_name=str(config["experiment_name"]),
            registered_model_name=str(config["registered_model_name"]),
        )

    def start(self, *, run_name: str | None = None) -> None:
        self.run = mlflow.start_run(run_name=run_name)

    @property
    def run_id(self) -> str:
        if self.run is None:
            raise RuntimeError("MLflow run has not been started.")
        return self.run.info.run_id

    def log_params(self, params: dict[str, Any]) -> None:
        mlflow.log_params({key: str(value) for key, value in params.items()})

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_pytorch_model(self, model: Any) -> None:
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name=self.registered_model_name,
        )

    def finish(self) -> None:
        if self.run is not None:
            mlflow.end_run()
            self.run = None
