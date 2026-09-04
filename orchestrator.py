import sys
import argparse
from pathlib import Path
from tqdm import tqdm

# FIX: Assuming orchestrator.py is in the root folder, we only need .parent
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.config import CONFIG
from src.features.pe_extractor import PEFeatureExtractor
from src.graph.feature_graph import FeatureGraphBuilder
from baselines.mlp.train_mlp import train_baseline_mlp
from baselines.lightgmb.train_lgbm import train_baseline_lgbm
# from src.models.gnn import MyGraphNeuralNetwork 

def run_pe_extraction():
    """
    Executes the feature extraction and graph building pipeline.
    """
    print("\n--- Starting PE Feature Extraction ---")
    extractor = PEFeatureExtractor()
    builder = FeatureGraphBuilder()
    
    samples_dir = project_root / CONFIG["PEsamples_path"]
    exe_files = list(samples_dir.glob("*.exe"))
    
    if not exe_files:
        print(f"No .exe files found in {samples_dir}")
        return

    # Process all files with a progress bar
    for file_path in tqdm(exe_files, desc="Processing PE files"):
        try:
            features = extractor.extract(str(file_path))
            graph = builder.build(features)
            # You can add logic here to save the graph to disk (CONFIG["graphs_path"])
        except Exception as e:
            tqdm.write(f"Failed to process {file_path.name}: {e}")
            
    print("--- Extraction Complete ---\n")


def main():
    # 1. Setup the Argument Parser
    parser = argparse.ArgumentParser(description="Malware Detection Thesis Orchestrator")
    parser.add_argument(
        "--task", 
        type=str, 
        required=True, 
        choices=["extract_pe", "train_mlp", "train_lgbm", "train_gnn", "train_gan"], 
        help="The pipeline task you want to execute."
    )
    
    args = parser.parse_args()

    # 2. Task Routing
    if args.task == "extract_pe":
        run_pe_extraction()
        
    elif args.task == "train_mlp":
        print("\n--- Starting Baseline MLP Training ---")
        train_baseline_mlp()
    
    elif args.task == "train_lgbm":
        print("\n--- Starting Baseline LGBM Training ---")
        train_baseline_lgbm()
        
    elif args.task == "train_gnn":
        print("\n--- [Placeholder] Starting GNN Training ---")
        # You will integrate src.training.train_gnn here in Weeks 5-6
        pass
        
    elif args.task == "train_gan":
        print("\n--- [Placeholder] Starting GAN Robustness Training ---")
        # You will integrate src.training.train_gan here in Weeks 7-8
        pass

if __name__ == "__main__":
    main()