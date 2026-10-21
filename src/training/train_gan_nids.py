import sys
import warnings
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.training.nids_dataset import get_nids_splits
from src.training.gan_pipeline import run_gan_pipeline
from src.utils.config import CONFIG
from src.utils.logger import ExperimentLogger

warnings.filterwarnings("ignore")


def train_gan():
    data_dir = project_root / CONFIG["cic_graphs_path"]

    # Chronological per-scenario split (past / near-future / distant-future)
    train_loader, val_loader, test_loader = get_nids_splits(data_dir, batch_size=128, seed=CONFIG["random_seed"])

    logger = ExperimentLogger(experiment_name="nids_gan_training", config=CONFIG)

    return run_gan_pipeline(
        train_data=train_loader.dataset,
        val_loader=val_loader,
        test_loaders={"test": test_loader},
        input_dim=train_loader.dataset[0].x.shape[1],
        per_node_range=False,  # flows are interchangeable: one range per feature
        lr=CONFIG["nids"]["learning_rate"],
        logger=logger,
    )


if __name__ == "__main__":
    train_gan()
