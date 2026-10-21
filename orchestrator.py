"""
Single entry point for the whole pipeline: python orchestrator.py -t <task>

Typical order:
  build_cicids / ember_graph               -> graphs in data/graphs/
  train_pe_gnn, train_nids_gnn             -> MFGraph detectors
  train_pe_gan, train_nids_gan             -> GAN-augmented detectors + PE generator
  train_pe_baselines, train_nids_baselines -> classical ML on pooled graph features
  eval_drift, eval_pe_adversarial, eval_nids_adversarial, unified_pipeline, plot_metrics
Every task writes its artefacts under experiments/ (see notebooks/ for the analysis).
"""
import sys
from pathlib import Path
import torch

project_root = Path(__file__).resolve().parents[0]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.config import CONFIG
from src.graph.ember_graph import ember_graph_building

def run_pe_extraction():
    """
    Builds PE graphs from the raw binaries in CONFIG["PEsamples_path"] (label 1 if the
    path contains a 'malware' folder). Used for demo samples; EMBER graphs come from ember_graph.
    """
    from tqdm import tqdm
    from src.features.pe_extractor import PEFeatureExtractor
    from src.graph.feature_graph import FeatureGraphBuilder
    print("\n--- Starting PE Feature Extraction ---")
    extractor = PEFeatureExtractor()
    builder = FeatureGraphBuilder()
    
    samples_dir = project_root / CONFIG["PEsamples_path"]

    # Define where the .pt files will be saved
    graphs_out_dir = project_root / CONFIG["ember_graphs_path"]
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
    from src.training.train_gnn_pe import train_gnn 

    graphs_dir = project_root / CONFIG["ember_graphs_path"]
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
    from src.training.train_gan_pe import train_gan
    
    graphs_dir = project_root / CONFIG["ember_graphs_path"]
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
    print("\n========================================================")
    print("      EVALUATION METRICS & COMPARISON VISUALIZER")
    print("========================================================")
    from datetime import datetime
    from src.evaluation.plot_metrics import generate_pe_comparison, generate_nids_comparison

    print("\nSelect evaluation domain:")
    print("  1. PE Domain (Host / EMBER)")
    print("  2. NIDS Domain (Network / Flows)")
    print("  3. Both Domains (Default)")

    domain_choice = input("\nEnter your choice (1/2/3) [Default: 3]: ").strip()
    if not domain_choice:
        domain_choice = '3'

    print("\nWhat would you like to generate?")
    print("  1. Tables only (CSV summaries)")
    print("  2. Graphs only (PNG bar charts)")
    print("  3. Both (Default)")

    out_choice = input("\nEnter your choice (1/2/3) [Default: 3]: ").strip()
    if not out_choice:
        out_choice = '3'

    gen_tables = out_choice in ('1', '3')
    gen_plots = out_choice in ('2', '3')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_exp_dir = project_root / CONFIG["experiments_path"]
    save_dir = base_exp_dir / "model_comparisons" / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Info] Output directory: {save_dir}")

    if domain_choice in ('1', '3'):
        print("\n--- Processing PE Host Domain ---")
        generate_pe_comparison(save_dir, base_exp_dir, gen_plots=gen_plots, gen_tables=gen_tables)

    if domain_choice in ('2', '3'):
        print("\n--- Processing NIDS Network Domain ---")
        generate_nids_comparison(save_dir, base_exp_dir, gen_plots=gen_plots, gen_tables=gen_tables)

    print(f"\n[Success] Visualization run completed. All artifacts saved to:\n  {save_dir}")

def drift_task():
    print("\n========================================================")
    print("           CONCEPT DRIFT EVALUATION PIPELINE           ")
    print("========================================================")
    from src.evaluation.concept_drift import (
        evaluate_pe_concept_drift,
        evaluate_nids_concept_drift,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Running on device: {device}")

    print("\nSelect concept drift domain:")
    print("  1. PE Domain (Host / EMBER Monthly Drift)")
    print("  2. NIDS Domain (Network / Flow Chunks Drift)")
    print("  3. Both Domains (Default)")

    choice = input("\nEnter your choice (1/2/3) [Default: 3]: ").strip()
    if not choice:
        choice = "3"

    if choice in ("1", "3"):
        try:
            evaluate_pe_concept_drift(device)
        except Exception as e:
            print(f"[Error] PE concept drift evaluation failed: {e}")

    if choice in ("2", "3"):
        try:
            evaluate_nids_concept_drift(device)
        except Exception as e:
            print(f"[Error] NIDS concept drift evaluation failed: {e}")

    print("\n[Success] Concept drift task completed.")

def build_cicids_graphs():
    from src.graph.cic_graph import build_cicids_graphs
    graphs_dir = project_root / CONFIG["cic_graphs_path"]
    raw_csv_path = project_root / "data" / "raw" / "CIC-IDS-2017" 
    out_dir = graphs_dir
    
    if not raw_csv_path.exists():
        print(f"[Error] CIC-IDS2017 CSV not found at {raw_csv_path}")
    else:
        csv_files = list(raw_csv_path.glob("*.csv"))
        print(f"[Info] Found {len(csv_files)} CSV files to process.")
        
        for csv_file in csv_files:
            print(f"\n--- Processing {csv_file.name} ---")
            build_cicids_graphs(str(csv_file), str(out_dir), window_size=500)



def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Malware Detection Thesis Orchestrator",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    tasks = [
        "extract_pe", "ember_graph", "train_pe_gnn", "train_pe_gan", "train_nids_gan", "train_nids_gnn", 
        "train_nids_baselines", "train_pe_baselines", "plot_metrics", "eval_drift", 
        "build_cicids", "eval_nids_adversarial", "eval_pe_adversarial", "unified_pipeline"
    ]
    
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

    elif args.task == "train_pe_gnn":
        gnn_task()
        
    elif args.task == "train_pe_gan":
        gan_task()

    elif args.task == "train_nids_gan":
        from src.training.train_gan_nids import train_gan as train_gan_nids
        train_gan_nids()

    elif args.task == "train_pe_baselines":
        from baselines.train_baseline_pe import train_pe_baselines
        train_pe_baselines()

    elif args.task == "build_cicids":
        build_cicids_graphs()

    elif args.task == "train_nids_gnn":
        from src.training.train_gnn_nids import train_nids_gnn
        train_nids_gnn()

    elif args.task == "train_nids_baselines":
        from baselines.train_baseline_nids import train_and_save_baselines
        train_and_save_baselines() 

    elif args.task == "eval_pe_adversarial":
        from src.evaluation.pe_adversarial import evaluate_pe_adversarial
        evaluate_pe_adversarial()

    elif args.task == "eval_nids_adversarial":
        from src.evaluation.nids_adversarial import evaluate_nids_adversarial
        evaluate_nids_adversarial() 

    elif args.task == "plot_metrics":
        plot_metrics_task()

    elif args.task == "eval_drift":
        drift_task()

    # --- UNIFIED PIPELINE TASK ---
    elif args.task == "unified_pipeline":
        from src.evaluation.unified_pipeline import evaluate_unified_pipeline
        evaluate_unified_pipeline()

if __name__ == "__main__":
    main()