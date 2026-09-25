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
    Parses EMBER .jsonl files to build the training graph dataset.
    This bypasses PE extraction since EMBER provides pre-extracted LIEF features.
    """
    builder = FeatureGraphBuilder()
    parser = EmberFeatureParser() 
    
    # Paths configured in your CONFIG dictionary
    ember_dir = project_root / CONFIG["ember_path"]
    graphs_out_dir = project_root / CONFIG["graphs_path"]
    graphs_out_dir.mkdir(parents=True, exist_ok=True)
    
    jsonl_files = list(ember_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No .jsonl files found in {ember_dir}. Please download and extract EMBER 2018.")
        return
        
    for jsonl_file in jsonl_files:

        # Quickly count the total lines to enable the percentage bar
        with open(jsonl_file, 'r') as f:
            total_lines = sum(1 for _ in f)

        # Counters for reporting
        saved_count = 0
        skipped_date = 0
        skipped_unlabeled = 0

        # Read the file line-by-line directly to prevent RAM exhaustion
        with open(jsonl_file, 'r') as f:
            # tqdm will now show processing speed (graphs/second) instead of a percentage
            for idx, line in enumerate(tqdm(f, total=total_lines, desc="Building Graphs")):
                try:
                    raw_dict = json.loads(line)
                    
                    label = raw_dict.get("label", -1)
                    if label == -1:
                        skipped_unlabeled += 1
                        continue
                    
                    # I recommend uncommenting this filter. It will instantly skip 
                    # 90% of the dataset, speeding up your run significantly.
                    appeared = raw_dict.get("appeared", "")
                    if "2018-01" not in appeared:
                         skipped_date += 1
                         continue
                    
                    features = parser.parse(raw_dict)
                    graph = builder.build(features, label=label)
                    
                    # Append the loop index (idx) so files never overwrite each other
                    sha256 = raw_dict.get("sha256", f"unknown_{idx}")
                    out_path = graphs_out_dir / f"{sha256}.pt"
                    
                    torch.save(graph, out_path)
                    saved_count += 1
                except Exception as e:
                    tqdm.write(f"Error on line {idx}: {e}")
                    # Remove the 'break' here so it doesn't stop the whole file on a single bad line
                    continue

        print(f"Finished {jsonl_file.name}:")
        print(f"  -> Saved: {saved_count}")
        print(f"  -> Skipped (Not Jan 2018): {skipped_date}")
        print(f"  -> Skipped (Unlabeled): {skipped_unlabeled}")
                
    print("\n--- EMBER Graph Construction Complete ---")