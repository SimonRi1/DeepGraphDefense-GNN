import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[0]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.config import CONFIG
from src.graph.ember_graph import ember_graph_building

def run_pe_extraction():
    import torch
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


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Malware Detection Thesis Orchestrator",
        formatter_class=argparse.RawTextHelpFormatter
    )
    tasks = ["extract_pe", "ember_graph", "train_mlp", "train_lgbm", "train_gnn", "train_gan", "plot_metrics"]
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
        print("\n--- Starting GNN Training ---")
        from src.training.train_gnn import train_gnn 

        graphs_dir = project_root / CONFIG["graphs_path"]
        train_dir = graphs_dir / "train"
        test_dir = graphs_dir / "test"
        
        print(f"[Info] Checking directories: {train_dir} and {test_dir}...")
        
        # Check if at least one .pt file exists in both subfolders
        has_train_graphs = next(train_dir.glob("*.pt"), None) is not None
        has_test_graphs = next(test_dir.glob("*.pt"), None) is not None
        
        if not (has_train_graphs and has_test_graphs):
            print("[Warning] Chronological training/test graphs not found. Automatically building from EMBER .jsonl files...")
            ember_graph_building() 
        else:
            print("[Info] Found existing chronologically sorted graphs. Proceeding to training.")
            
        train_gnn()
        
    elif args.task == "train_gan":
        print("\n--- Starting GAN Robustness & MFGraph Training ---")
        from src.training.train_gan import train_gan
        
        graphs_dir = project_root / CONFIG["graphs_path"]
        train_dir = graphs_dir / "train"
        test_dir = graphs_dir / "test"
        
        print(f"[Info] Checking directories: {train_dir} and {test_dir}...")

        # Check if at least one .pt file exists in both subfolders
        has_train_graphs = next(train_dir.glob("*.pt"), None) is not None
        has_test_graphs = next(test_dir.glob("*.pt"), None) is not None
        
        if not (has_train_graphs and has_test_graphs):
            print("[Warning] Training graphs not found. Automatically building from EMBER .jsonl files...")
            ember_graph_building()
        else:
            print("[Info] Found existing training graphs. Proceeding to GAN pipeline.")
            
        train_gan()

    elif args.task == "plot_metrics":
        print("\n--- Starting Models Visualization ---")
        from datetime import datetime
        from src.utils.plot_metrics import plot_comparisons, generate_comparison_table
        
        available_models = ["mlp_baseline", "lgbm_baseline", "gnn_training", "gan_training"]
        
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


if __name__ == "__main__":
    main()