# Evidently and Prometheus Monitoring

Issue #44 exposes the existing drift scorecard through two reproducible monitoring paths.

## 1. Generate actual monitoring datasets

First generate the deterministic drift scorecard:

```powershell
python -m scripts.run_drift_scorecard
```

Then materialize the actual reference/current rows used by Evidently:

```powershell
python -m scripts.prepare_monitoring_data
```

Expected files:

- `reports/monitoring/reference_monitoring.csv`
- `reports/monitoring/current_monitoring.csv`
- `reports/monitoring/drift_scorecard.json`
- `reports/monitoring/drift_scorecard.csv`

The CSV rows contain image-derived statistics and model confidence/prediction results. No monitoring values are hard-coded.

## 2. Generate the Evidently report

Install the project dependencies, then run:

```powershell
python -m scripts.run_evidently_report
```

The HTML report is written to:

```text
reports/monitoring/evidently_drift.html
```

The report compares the actual clean reference dataset with the actual current corrupted dataset.

## 3. Start the Prometheus exporter

In a second terminal:

```powershell
python -m scripts.export_prometheus
```

The exporter serves:

```text
http://127.0.0.1:8000/metrics
```

Prometheus can scrape that endpoint with:

```yaml
scrape_configs:
  - job_name: pet-breed-monitoring
    static_configs:
      - targets: ["127.0.0.1:8000"]
```

The exporter publishes:

- `pet_breed_inference_requests_total`
- `pet_breed_inference_errors_total`
- `pet_breed_inference_latency_seconds`
- `pet_breed_inference_confidence`
- `pet_breed_inference_abstentions_total`
- `pet_breed_drift_score{detector=...}`
- `pet_breed_drift_flag{detector=...}`

The drift metrics are loaded directly from the generated scorecard. Request, error, latency, confidence, and abstention metrics are also available through the reusable instrumentation helpers for the serving process.

## Reproducibility and data integrity

The reference is the clean test split already used by the monitoring scorecard. The current dataset is derived from the existing deterministic corruption suite. The protected test images are read only; they are never overwritten or modified.

Do not manually enter monitoring numbers into reports. Regenerate the CSV, HTML, and metrics from the actual local run.
