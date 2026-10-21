import csv
import json
from datetime import datetime
from pathlib import Path


class ExperimentLogger:
    """
    Creates experiments/<experiment_name>/<YYYYmmdd_HHMMSS>/ for one run, saves the
    config there and appends metrics to CSV files as training progresses
    (rows are flushed every epoch, so a crashed run still keeps its history).
    """
    def __init__(self, experiment_name: str, config: dict, base_dir: str = "experiments"):
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(base_dir) / experiment_name / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_log = []

        with open(self.run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"[Logger] Run saved to: {self.run_dir}")

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
        """One row per evaluation period (EMBER month or NIDS test split) -> concept_drift.csv"""
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

    def log_gan_epoch(self, epoch: int, metrics: dict):
        """
        Track WGAN-GP metrics per epoch.
        w_dist (critic's Wasserstein estimate) should stay bounded: steady growth means
        the critics win and the generator is not catching up;
        feat_mean_gap is the mean absolute gap between real and fake feature means.
        """
        entry = {"epoch": epoch, **{k: round(v, 4) for k, v in metrics.items()}}

        path = self.run_dir / "gan_metrics.csv"
        write_header = not path.exists()

        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=entry.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(entry)
