import torch
import json
from tqdm import tqdm
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.config import CONFIG
from src.graph.feature_graph import FeatureGraphBuilder
from src.features.pe_extractor import EmberFeatureParser

def ember_graph_building():
    """
    Parses EMBER .jsonl files to build the training and testing graph datasets.
    Dynamically routes graphs based on the 'appeared' month to evaluate concept drift.
    - 2018-01 goes to data/graphs/train/
    - 2018-02 through 2018-12 go to data/graphs/test/02/, data/graphs/test/03/, etc.
    """
    builder = FeatureGraphBuilder()
    parser = EmberFeatureParser() 
    
    # Base directory setup
    ember_dir = project_root / CONFIG["ember_path"]
    graphs_out_dir = project_root / CONFIG["graphs_path"]
    
    train_out_dir = graphs_out_dir / "train"
    test_out_dir = graphs_out_dir / "test"
    train_out_dir.mkdir(parents=True, exist_ok=True)
    test_out_dir.mkdir(parents=True, exist_ok=True)
    
    jsonl_files = list(ember_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No .jsonl files found in {ember_dir}. Please download and extract EMBER 2018.")
        return
        
    for jsonl_file in jsonl_files:
        # Count lines for the progress bar
        with open(jsonl_file, 'r') as f:
            total_lines = sum(1 for _ in f)

        saved_train = 0
        saved_test = 0
        skipped_date = 0
        skipped_unlabeled = 0

        print(f"\nProcessing {jsonl_file.name}...")
        with open(jsonl_file, 'r') as f:
            for idx, line in enumerate(tqdm(f, total=total_lines, desc=f"Building Graphs")):
                try:
                    raw_dict = json.loads(line)
                    
                    label = raw_dict.get("label", -1)
                    if label == -1:
                        skipped_unlabeled += 1
                        continue
                    
                    appeared = raw_dict.get("appeared", "")
                    
                    # Ensure the sample is from 2018 and has a valid month
                    if not appeared or not appeared.startswith("2018-"):
                        skipped_date += 1
                        continue
                        
                    month = appeared.split("-")[1] # Extracts "01", "02", ..., "12"
                    
                    # Concept Drift Routing Logic
                    if month == "01":
                        # January is our baseline training set
                        target_dir = train_out_dir
                        saved_train += 1
                    else:
                        # February to December are our chronological test sets
                        target_dir = test_out_dir / month
                        target_dir.mkdir(parents=True, exist_ok=True)
                        saved_test += 1
                        
                    features = parser.parse(raw_dict)
                    graph = builder.build(features, label=label)
                    
                    sha256 = raw_dict.get("sha256", f"unknown_{idx}")
                    out_path = target_dir / f"{sha256}.pt"
                    
                    torch.save(graph, out_path)
                    
                except Exception as e:
                    tqdm.write(f"Error on line {idx}: {e}")
                    continue

        print(f"Finished {jsonl_file.name}:")
        print(f"  -> Saved in Train (Jan): {saved_train}")
        print(f"  -> Saved in Test (Feb-Dec): {saved_test}")
        print(f"  -> Skipped (Not 2018): {skipped_date}")
        print(f"  -> Skipped (Unlabeled): {skipped_unlabeled}")
                
    print("\n--- EMBER Graph Construction Complete ---")