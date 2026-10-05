import sys
from pathlib import Path
import torch

project_root = Path(__file__).resolve().parents[0]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.config import CONFIG
from src.graph.ember_graph import ember_graph_building

def run_pe_extraction():
    from tqdm import tqdm
    from src.features.pe_extractor import PEFeatureExtractor
    from src.graph.feature_graph import FeatureGraphBuilder
    """
    Executes the feature extraction and graph building pipeline.
    """
    print("\n--- Starting PE Feature Extraction ---")
    extractor = PEFeatureExtractor()
    builder = FeatureGraphBuilder()
    
    samples_dir = project_root / CONFIG["PEsamples_path"]

    # Define where the .pt files will be saved
    graphs_out_dir = project_root / CONFIG["graphs_path"]
    graphs_out_dir.mkdir(parents=True, exist_ok=True)

    exe_files = list(samples_dir.glob("**/*.exe"))
    
    if not exe_files:
        print(f"No .exe files found in {samples_dir}")
        return

    for file_path in tqdm(exe_files, desc="Processing PE files"):
        try:
            features = extractor.extract(str(file_path))

            if features is None:
                continue
            
            label = 1 if "malware" in file_path.parts else 0
            graph = builder.build(features, label=label)
            
            out_path = graphs_out_dir / f"{file_path.stem}.pt"
            torch.save(graph, out_path)

        except Exception as e:
            tqdm.write(f"Failed to process {file_path.name}: {e}")
            
    print("--- Extraction Complete ---\n")

def gnn_task():
    print("\n--- Starting GNN Training ---")
    from src.training.train_gnn import train_gnn 

    graphs_dir = project_root / CONFIG["graphs_path"]
    train_dir = graphs_dir / "train"
    test_dir = graphs_dir / "test"
    
    print(f"[Info] Checking directories: {train_dir} and {test_dir}...")
    
    # Use rglob() to search inside subfolders as well (e.g., test/02, test/03)
    has_train_graphs = next(train_dir.rglob("*.pt"), None) is not None
    has_test_graphs = next(test_dir.rglob("*.pt"), None) is not None
    
    if not (has_train_graphs and has_test_graphs):
        print("[Warning] Chronological training/test graphs not found. Automatically building from EMBER .jsonl files...")
        ember_graph_building() 
    else:
        print("[Info] Found existing chronologically sorted graphs. Proceeding to training.")
        
    train_gnn()

def gan_task():
    print("\n--- Starting GAN Robustness & MFGraph Training ---")
    from src.training.train_gan import train_gan
    
    graphs_dir = project_root / CONFIG["graphs_path"]
    train_dir = graphs_dir / "train"
    test_dir = graphs_dir / "test"
    
    print(f"[Info] Checking directories: {train_dir} and {test_dir}...")

    # Check if at least one .pt file exists in both subfolders
    has_train_graphs = next(train_dir.rglob("*.pt"), None) is not None
    has_test_graphs = next(test_dir.rglob("*.pt"), None) is not None
    
    if not (has_train_graphs and has_test_graphs):
        print("[Warning] Training graphs not found. Automatically building from EMBER .jsonl files...")
        ember_graph_building()
    else:
        print("[Info] Found existing training graphs. Proceeding to GAN pipeline.")
        
    train_gan()

def plot_metrics_task():
    print("\n--- Starting Models Visualization ---")
    from datetime import datetime
    from src.evaluation.plot_metrics import plot_comparisons, generate_comparison_table
    
    available_models = ["mlp_training", "lgbm_training", "gnn_training", "gan_training"]
    
    print("Available models:")
    for i, model in enumerate(available_models, 1):
        print(f"  {i}. {model}")
    print(f"  {len(available_models) + 1}. All of the above (Default)")
    
    choice = input("\nEnter the numbers of the models to compare (e.x: 1,2) [Default: 5]: ").strip()
    
    if not choice or str(len(available_models) + 1) in choice:
        selected_models = available_models
    else:
        try:
            indices = [int(idx.strip()) - 1 for idx in choice.split(',')]
            selected_models = [available_models[i] for i in indices if 0 <= i < len(available_models)]
        except ValueError:
            print("[Error] Invalid input. Please enter numbers separated by commas.")
            return
            
    if not selected_models:
        print("[Error] No valid models selected.")
        return
        
    print(f"\n[Info] Selected for comparison: {', '.join(selected_models)}")
    
    print("\nWhat would you like to generate?")
    print("  1. Table only (AUC, F1, Accuracy)")
    print("  2. Graphs only (AUC and Loss curves)")
    print("  3. Both (Default)")
    
    out_choice = input("\nEnter your choice (1/2/3) [Default: 3]: ").strip()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = project_root / CONFIG["experiments_path"] / "model_comparisons" / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[Info] Saving outputs to: {save_dir}")
    base_exp_dir = project_root / CONFIG["experiments_path"]
    
    if out_choice == '1':
        generate_comparison_table(selected_models, save_dir, base_exp_dir)
        print(f"\n[Success] Comparison Table saved in {save_dir}")
    elif out_choice == '2':
        plot_comparisons(selected_models, save_dir, base_exp_dir)
        print(f"\n[Success] Plots saved in {save_dir}")
    else:
        plot_comparisons(selected_models, save_dir, base_exp_dir)
        generate_comparison_table(selected_models, save_dir, base_exp_dir)
        print(f"\n[Success] Plots and Comparison Table saved in {save_dir}")

def drift_task():
    print("\n--- Starting Concept Drift Evaluation ---")
    from src.evaluation.concept_drift import evaluate_concept_drift
    from src.training.pe_dataset import PEGraphDataset
    import os
    import torch
    from torch_geometric.loader import DataLoader
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device} for evaluation")

    # Maximize CPU cores for reading the thousands of .pt files
    max_cores = os.cpu_count() or 6
    optimal_workers = min(8, max_cores - 1)
    optimal_workers = max(0, optimal_workers)
    
    # INCREASE BATCH SIZE FOR EVALUATION TO SPEED UP INFERENCE
    eval_batch_size = 512 
    
    gnn_exp_dir = project_root / CONFIG["experiments_path"] / "gnn_training"
    gan_exp_dir = project_root / CONFIG["experiments_path"] / "gan_training"
    mlp_exp_dir = project_root / CONFIG["experiments_path"] / "mlp_training"
    lgbm_exp_dir = project_root / CONFIG["experiments_path"] / "lgbm_training"
    
    try:
        gnn_weights = max(gnn_exp_dir.rglob("*.pth"), key=lambda f: f.stat().st_mtime)
        gan_weights = max(gan_exp_dir.rglob("*.pth"), key=lambda f: f.stat().st_mtime)
        mlp_weights = max(mlp_exp_dir.rglob("*.pth"), key=lambda f: f.stat().st_mtime)
        lgbm_weights = max(lgbm_exp_dir.rglob("*.txt"), key=lambda f: f.stat().st_mtime)
    except ValueError as e: # FIXED: Added 'as e' to capture the error properly
        print("[Error] Unable to find model weights. Train all 4 models first.")
        print(f"Details: {e}")
        return
    
    # Ensure ALL 4 weights actually exist on disk
    if not all([gnn_weights.exists(), gan_weights.exists(), mlp_weights.exists(), lgbm_weights.exists()]):
        print("[Error] One or more model weights are missing. Train all 4 models first.")
        return

    print("[Info] Loading monthly EMBER datasets...")
    monthly_loaders = {}
    
    test_dir = project_root / CONFIG["graphs_path"] / "test"
    
    for month in range(2, 13):
        month_str = f"{month:02d}" # Formatta come "02", "03", ecc.
        month_path = test_dir / month_str
        
        # Changed to rglob to safely ensure .pt files exist inside the folder
        if month_path.exists() and next(month_path.rglob("*.pt"), None):
            dataset = PEGraphDataset(month_path)
            
            # TURBOCHARGED DATALOADER
            loader = DataLoader(
                dataset, 
                batch_size=eval_batch_size, 
                shuffle=False,
                num_workers=optimal_workers,
                persistent_workers=(optimal_workers > 0)
            )
            
            monthly_loaders[month] = loader
            print(f"  - Month {month_str}: Found {len(dataset)} samples.")
        else:
            print(f"  - Month {month_str}: No sample found.")
    
    if not monthly_loaders:
        print("[Error] No monthly data found in test/. Check the extraction.")
        return

    evaluate_concept_drift(
        gnn_weights_path=gnn_weights, 
        gan_weights_path=gan_weights, 
        mlp_weights_path=mlp_weights, 
        lgbm_weights_path=lgbm_weights, 
        monthly_loaders=monthly_loaders, 
        device=device
    )

def build_cicids_graphs():
    from src.training.nids_dataset import build_cicids_graphs
    # Adjust this path to point to whichever specific Friday/Thursday CSV you downloaded
    raw_csv_path = project_root / "data" / "raw" / "CIC-IDS-2017" 
    out_dir = project_root / "data" / "graphs" / "nids_graphs"
    
    if not raw_csv_path.exists():
        print(f"[Error] CIC-IDS2017 CSV not found at {raw_csv_path}")
    else:
        # Find all CSV files in the directory
            csv_files = list(raw_csv_path.glob("*.csv"))
            print(f"[Info] Found {len(csv_files)} CSV files to process.")
            
            for csv_file in csv_files:
                print(f"\n--- Processing {csv_file.name} ---")
                # I recommend window_size=500 to generate a huge dataset of graphs
                build_cicids_graphs(str(csv_file), str(out_dir), window_size=500)

def adversarial_eval():
    from src.evaluation.nids_adversarial import evaluate_robustness
    from src.models.gnn import MFGraph
    from src.training.pe_dataset import PEGraphDataset
    import os
    from torch_geometric.loader import DataLoader
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = PEGraphDataset(str(project_root / "data" / "graphs" / "nids_graphs"))
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    nids_exp_dir = project_root / CONFIG["experiments_path"] / "nids_gnn"
    gnn_weights = max(nids_exp_dir.rglob("*.pth"), key=lambda f: f.stat().st_mtime)
    
    # FIXED: Added dropout_rate from CONFIG
    model = MFGraph(
        input_dim=9, 
        hidden_dim=CONFIG["gnn"]["hidden_dim"], 
        k=CONFIG["gnn"]["k"],
        dropout_rate=CONFIG["gnn"].get("dropout_rate", 0.5)
    ).to(device)
    
    model.load_state_dict(torch.load(gnn_weights))
    
    evaluate_robustness(
        model, loader, device, 
        feature_idx=2, 
        scale_factors=[1.0, 0.8, 0.5, 0.1, 2.0, 5.0], 
        feature_name="Packet Size"
    )
    
    evaluate_robustness(
        model, loader, device, 
        feature_idx=4, 
        scale_factors=[1.0, 2.0, 5.0, 10.0], 
        feature_name="Inter-Arrival Time"
    )

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Malware Detection Thesis Orchestrator",
        formatter_class=argparse.RawTextHelpFormatter
    )
    tasks = ["extract_pe", "ember_graph", "train_mlp", "train_lgbm", "train_gnn", "train_gan", "plot_metrics", "evaluate_drift", "build_cicids", "train_nids", "eval_nids_adversarial", "eval_nids_baseline", "unified_pipeline"]
    parser.add_argument(
        "-t", "--task",
        type=str, 
        required=True, 
        choices=tasks,
        metavar="TASK",
        help=f"The pipeline task you want to execute.\nOptions: {', '.join(tasks)}"
    )
    
    args = parser.parse_args()

    if args.task == "extract_pe":
        run_pe_extraction()
    
    elif args.task == "ember_graph":
        print("\n--- Starting EMBER Graph Construction ---")
        ember_graph_building()
        
    elif args.task == "train_mlp":
        print("\n--- Starting Baseline MLP Training ---")
        from baselines.mlp.train_mlp import train_baseline_mlp
        train_baseline_mlp()
    
    elif args.task == "train_lgbm":
        print("\n--- Starting Baseline LGBM Training ---")
        from baselines.lightgbm.train_lgbm import train_baseline_lgbm
        train_baseline_lgbm()

    elif args.task == "train_gnn":
        gnn_task()
        
    elif args.task == "train_gan":
        gan_task()

    elif args.task == "plot_metrics":
        plot_metrics_task()

    elif args.task == "evaluate_drift":
        drift_task()

    elif args.task == "build_cicids":
        build_cicids_graphs()

    elif args.task == "train_nids":
        from src.training.train_nids import train_nids_gnn
        train_nids_gnn()

    elif args.task == "eval_nids_adversarial":
        adversarial_eval()

    elif args.task == "eval_nids_baseline":
        from baselines.nids_baseline.train_trad_nids import train_and_evaluate_baselines
        train_and_evaluate_baselines() 

    elif args.task == "unified_pipeline":
        from src.evaluation.unified_pipeline import evaluate_unified_pipeline
        evaluate_unified_pipeline()

if __name__ == "__main__":
    main()