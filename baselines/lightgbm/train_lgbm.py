import sys
import numpy as np
from pathlib import Path
import lightgbm as lgb
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

# Ensure the project root is in the Python path to allow absolute imports
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from baselines.mlp.dataset import EmberFlatDataset
from src.utils.logger import ExperimentLogger
from src.utils.config import CONFIG

def custom_eval_metrics(y_true, y_pred):
    # Convert LightGBM's continuous probabilities into binary 0/1 predictions
    y_bin = (y_pred > 0.5).astype(int)
    
    # Return format required by LightGBM: (metric_name, metric_value, is_higher_better)
    return [
        ('f1', f1_score(y_true, y_bin), True),
        ('accuracy', accuracy_score(y_true, y_bin), True),
        ('precision', precision_score(y_true, y_bin, zero_division=0), True),
        ('recall', recall_score(y_true, y_bin, zero_division=0), True)
    ]

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
        n_jobs=-1,  # -1 means use all available CPU cores
        verbose=-1,  # Add this line to silence the C++ backend for status bar
    )

    # 4. Training Phase
    print(f"\nStarting LightGBM training for {lgbm_config['n_estimators']} trees...")
    # Create the progress bar
    pbar = tqdm(
        total=lgbm_config["n_estimators"], 
        desc="Building Trees",
        file=sys.stdout,
        dynamic_ncols=True
        )
    # Define a custom callback that LightGBM triggers after every tree
    def tqdm_callback(env):
        pbar.update(1)
    # Train the model, passing our custom callback and silencing default text spam
    model.fit(
        X_train, y_train,
        # 1. Use eval_X and eval_y instead of eval_set to fix the deprecation warning
        #eval_X=[X_train, X_test],
        #eval_y=[y_train, y_test], 
        eval_set=[(X_train, y_train), (X_test, y_test)], 
        eval_names=['train', 'test'],
        # 2. Add the custom_eval_metrics function to the list
        eval_metric=["binary_logloss", "auc", custom_eval_metrics],         
        callbacks=[
            tqdm_callback, 
            lgb.log_evaluation(period=0)
        ]
    )
    # Close the progress bar when finished
    pbar.close()
        
    # 5. Extract and Log Per-Iteration Metrics (This makes the curves)
    print("\nExtracting per-iteration metrics for graphs...")
    results = model.evals_result_
    
    # Loop through every built tree ("epoch") and log its metrics to the CSV
    num_trees = len(results['train']['binary_logloss'])

    # Define how many points you want on your graph (matching the MLP epochs for compare)
    target_points = 20
    step_size = num_trees // target_points  # E.g., 1000 // 20 = 50

    for step in range(1, target_points + 1):
        i = (step * step_size) - 1
        # 1. Extract metrics
        epoch_metrics = {
            "train_loss": results['train']['binary_logloss'][i],
            "test_loss": results['test']['binary_logloss'][i],
            "auc": results['test']['auc'][i],
            "f1": results['test']['f1'][i],
            "accuracy": results['test']['accuracy'][i],
            "precision": results['test']['precision'][i],
            "recall": results['test']['recall'][i]
        }
        #logger.log_epoch(epoch=i+1, metrics=epoch_metrics)
        logger.log_epoch(epoch=step, metrics=epoch_metrics)  


    # 6. Calculate Metrics
    print("\nEvaluating final model performance...")
    preds_proba = model.predict_proba(X_test)[:, 1]
    preds_binary = model.predict(X_test)
    metrics = {
        "auc": roc_auc_score(y_test, preds_proba),
        "f1": f1_score(y_test, preds_binary),
        "accuracy": accuracy_score(y_test, preds_binary),
        "precision": precision_score(y_test, preds_binary),
        "recall": recall_score(y_test, preds_binary)
    }
    
    # Print a clean summary string to the terminal
    print(f"Epoch {(i+1):02d}/{num_trees} - "
          f"Train Loss: {epoch_metrics['train_loss']:.4f} | Test Loss: {epoch_metrics['test_loss']:.4f} | "
          f"AUC: {epoch_metrics['auc']:.4f} | F1: {epoch_metrics['f1']:.4f}")
    
    # Log metrics (using epoch=1 since LightGBM does the entire run at once)
    # logger.log_epoch(epoch=1, metrics=metrics)
        
    # 7. Save final model weights
    # LightGBM uses .txt format for saving its tree structures
    model_path = logger.run_dir / "lgbm_final.txt"
    model.booster_.save_model(str(model_path))
    
    print(f"\nTraining complete! Logs and model weights saved in:\n{logger.run_dir}")

if __name__ == "__main__":
    train_baseline_lgbm()