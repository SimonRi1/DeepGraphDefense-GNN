import sys
import time
from pathlib import Path

# Adjust path to find the src folder from the tests folder
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Ensure the import matches your actual file structure
from src.utils.logger import ExperimentLogger


def run_logger_test():
    print("Initializing Logger...")
    
    # 1. Test Initialization and Config Saving
    mock_config = {
        "model": "GraphNeuralNetwork",
        "learning_rate": 0.001,
        "epochs": 3
    }
    logger = ExperimentLogger(experiment_name="test_run", config=mock_config)

    # 2. Test Epoch Logging (CSV Append)
    print("Simulating training epochs...")
    for epoch in range(1, 4):
        mock_metrics = {
            "loss": 0.8 / epoch, 
            "accuracy": 0.7 + (0.05 * epoch)
        }
        logger.log_epoch(epoch=epoch, metrics=mock_metrics)
        print(f" Logged epoch {epoch}")
        time.sleep(0.5) 

    # 3. Test Concept Drift Logging
    print("Simulating concept drift logging...")
    logger.log_concept_drift(month="2026-09", metrics={"f1_score": 0.92, "drift": 0.01})
    logger.log_concept_drift(month="2026-10", metrics={"f1_score": 0.88, "drift": 0.05})

    print(f"\n Logger test complete! Please check the contents of:\n{logger.run_dir}")

if __name__ == "__main__":
    run_logger_test()