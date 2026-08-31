import csv
import json
from datetime import datetime
from pathlib import Path


class ExperimentLogger:
    def __init__(self, experiment_name: str, config: dict,
                 base_dir: str = "experiments"):
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(base_dir) / experiment_name / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_log = []

        with open(self.run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"[Logger] Run salvata in: {self.run_dir}")

    def log_epoch(self, epoch: int, metrics: dict):
        entry = {"epoch": epoch, **metrics}
        self.metrics_log.append(entry)
        
        path = self.run_dir / "metrics.csv"
        write_header = not path.exists()

        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=entry.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(entry)

    def log_concept_drift(self, month: str, metrics: dict):
        """Record the performance degradation every month"""
        entry = {"month": month, **metrics}
        path = self.run_dir / "concept_drift.csv"
        write_header = not path.exists()

        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=entry.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(entry)

    def save_model_info(self, info: dict):
        with open(self.run_dir / "model_info.json", "w") as f:
            json.dump(info, f, indent=2)
