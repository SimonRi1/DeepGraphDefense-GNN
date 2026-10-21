import sys
import warnings
from pathlib import Path
import torch
from torch.utils.data import random_split

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.training.pe_dataset import PEGraphDataset
from src.training.gan_pipeline import run_gan_pipeline, make_loader
from src.utils.config import CONFIG
from src.utils.logger import ExperimentLogger

gnn_config = CONFIG["gnn"]
warnings.filterwarnings("ignore")


def train_gan():
    graphs_dir = project_root / CONFIG["ember_graphs_path"]
    batch_size = gnn_config["batch_size"]

    print("\n[Data] Loading January 2018 dataset (Training set)...")
    jan_dataset = PEGraphDataset(graphs_dir / "train")

    # 80/20 Split: Train vs. Jan Validation
    train_size = int(0.8 * len(jan_dataset))
    val_size = len(jan_dataset) - train_size
    generator = torch.Generator().manual_seed(CONFIG["random_seed"])
    train_data, val_data = random_split(jan_dataset, [train_size, val_size], generator=generator)
    val_loader = make_loader(val_data, batch_size, shuffle=False)

    # Feb-Dec 2018: one test loader per month to measure concept drift
    test_loaders = {
        month_dir.name: make_loader(PEGraphDataset(month_dir), batch_size, shuffle=False)
        for month_dir in sorted((graphs_dir / "test").iterdir()) if month_dir.is_dir()
    }

    logger = ExperimentLogger(experiment_name="gan_training", config=CONFIG)

    return run_gan_pipeline(
        train_data=train_data,
        val_loader=val_loader,
        test_loaders=test_loaders,
        input_dim=jan_dataset[0].x.shape[1],
        per_node_range=True,  # node i is always the same PE feature group
        lr=gnn_config["learning_rate"],
        logger=logger,
    )


if __name__ == "__main__":
    train_gan()
