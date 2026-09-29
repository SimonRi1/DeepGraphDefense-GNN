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
    Automatically routes graphs into data/graphs/train and data/graphs/test.
    """
    builder = FeatureGraphBuilder()
    parser = EmberFeatureParser() 
    
    # 1. Base directory setup
    ember_dir = project_root / CONFIG["ember_path"]
    graphs_out_dir = project_root / CONFIG["graphs_path"]
    
    # Subdirectories for chronological split
    train_out_dir = graphs_out_dir / "train"
    test_out_dir = graphs_out_dir / "test"
    train_out_dir.mkdir(parents=True, exist_ok=True)
    test_out_dir.mkdir(parents=True, exist_ok=True)
    
    jsonl_files = list(ember_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No .jsonl files found in {ember_dir}. Please download and extract EMBER 2018.")
        return
        
    for jsonl_file in jsonl_files:
        # Determine whether the file belongs to train or test
        is_test_file = "test" in jsonl_file.name.lower()
        target_dir = test_out_dir if is_test_file else train_out_dir
        subset_name = "TEST" if is_test_file else "TRAIN"

        # Count lines for the progress bar
        with open(jsonl_file, 'r') as f:
            total_lines = sum(1 for _ in f)

        saved_count = 0
        skipped_date = 0
        skipped_unlabeled = 0

        print(f"\nProcessing {jsonl_file.name} [{subset_name}]...")
        with open(jsonl_file, 'r') as f:
            for idx, line in enumerate(tqdm(f, total=total_lines, desc=f"Building {subset_name} Graphs")):
                try:
                    raw_dict = json.loads(line)
                    
                    label = raw_dict.get("label", -1)
                    if label == -1:
                        skipped_unlabeled += 1
                        continue
                    
                    appeared = raw_dict.get("appeared", "")
                    
                    # Date Filtering:
                    # If replicating the paper's Jan 2018 training set: apply only to train files[cite: 1].
                    # For the test set (or concept drift evaluation), test files contain later dates (e.g., Nov-Dec 2018).
                    if not is_test_file:
                        if "2018-01" not in appeared:
                            skipped_date += 1
                            continue
                    
                    features = parser.parse(raw_dict)
                    graph = builder.build(features, label=label)
                    
                    sha256 = raw_dict.get("sha256", f"unknown_{idx}")
                    out_path = target_dir / f"{sha256}.pt"
                    
                    torch.save(graph, out_path)
                    saved_count += 1
                    
                except Exception as e:
                    tqdm.write(f"Error on line {idx}: {e}")
                    continue

        print(f"Finished {jsonl_file.name}:")
        print(f"  -> Saved in {target_dir.name}/: {saved_count}")
        print(f"  -> Skipped (Date Filter): {skipped_date}")
        print(f"  -> Skipped (Unlabeled): {skipped_unlabeled}")
                
    print("\n--- EMBER Graph Construction Complete ---")