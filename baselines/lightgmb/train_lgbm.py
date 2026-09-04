import sys
import numpy as np
from pathlib import Path
import lightgbm as lgb
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

# Ensure the project root is in the Python path to allow absolute imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from baselines.mlp.dataset import EmberFlatDataset
from src.utils.logger import ExperimentLogger
from src.utils.config import CONFIG

def train_baseline_lgbm():
    # 1. Initialize Logger
    # We pass the entire CONFIG so the logger saves all hyperparameters for reproducibility
    logger = ExperimentLogger(experiment_name="lgbm_baseline", config=CONFIG)

    # 2. Load Data
    # Resolves to thesis-project/data/raw/ember2018 using the config path
    data_dir = CONFIG["ember_path"]
    
    print("Loading datasets into memory (this may take a moment)...")
    train_dataset = EmberFlatDataset(data_dir=str(data_dir), split="train")
    test_dataset = EmberFlatDataset(data_dir=str(data_dir), split="test")

    # For LightGBM, we don't need DataLoaders. We extract the raw NumPy arrays directly.
    X_train, y_train = train_dataset.X, train_dataset.y
    X_test, y_test = test_dataset.X, test_dataset.y

    # Extract LightGBM specific parameters from config
    lgbm_config = CONFIG["lgbm"]

    # 3. Initialize Model
    model = lgb.LGBMClassifier(
        n_estimators=lgbm_config["n_estimators"],
        learning_rate=lgbm_config["learning_rate"],
        num_leaves=lgbm_config["num_leaves"],
        objective=lgbm_config["objective"],
        random_state=CONFIG.get("random_seed", 42),
        n_jobs=-1  # -1 means use all available CPU cores
    )

    # 4. Training Phase
    print(f"\nStarting LightGBM training for {lgbm_config['n_estimators']} trees...")
    # Create the progress bar
    pbar = tqdm(total=lgbm_config["n_estimators"], desc="Building Trees")
    # Define a custom callback that LightGBM triggers after every tree
    def tqdm_callback(env):
        pbar.update(1)
    # Train the model, passing our custom callback and silencing default text spam
    model.fit(
        X_train, y_train,
        eval_X = X_test,
        eval_y = y_test,
        callbacks=[
            tqdm_callback, 
            lgb.log_evaluation(period=0)  # This hides LightGBM's default text logs so the bar looks clean
        ]
    )
    # Close the progress bar when finished
    pbar.close()
        
    # 5. Evaluation Phase
    print("\nEvaluating on test set...")
    # Get probabilities [0, 1] for AUC, and binary 0/1 for F1, Accuracy, etc.
    preds_proba = model.predict_proba(X_test)[:, 1]
    preds_binary = model.predict(X_test)
    
    # 6. Calculate Metrics
    metrics = {
        "auc": roc_auc_score(y_test, preds_proba),
        "f1": f1_score(y_test, preds_binary),
        "accuracy": accuracy_score(y_test, preds_binary),
        "precision": precision_score(y_test, preds_binary),
        "recall": recall_score(y_test, preds_binary)
    }
    
    # Print a clean summary string to the terminal
    print(f"Final Results - "
          f"AUC: {metrics['auc']:.4f} | F1: {metrics['f1']:.4f} | "
          f"Accuracy: {metrics['accuracy']:.4f}")
    
    # Log metrics (using epoch=1 since LightGBM does the entire run at once)
    logger.log_epoch(epoch=1, metrics=metrics)
        
    # 7. Save final model weights
    # LightGBM uses .txt format for saving its tree structures
    model_path = logger.run_dir / "lgbm_final.txt"
    model.booster_.save_model(str(model_path))
    
    print(f"\nTraining complete! Logs and model weights saved in:\n{logger.run_dir}")

if __name__ == "__main__":
    train_baseline_lgbm()