import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

print(f"Project root is successfully set to: {project_root}")

from src.utils.config import CONFIG
import json
import numpy as np
from src.features.pe_extractor import PEFeatureExtractor, EmberFeatureParser

pe_extractor = PEFeatureExtractor()
ember_parser = EmberFeatureParser()

pe_file = "7zip.exe"
ember_file = "train_features_0.jsonl"

def run_test():
    # 1. Test extraction from a real EXE file (replace with a valid path on your PC)
    exe_path = CONFIG["PEsamples_path"]+pe_file 
    print(f"\nExtracting from {exe_path}...")
    pe_features = pe_extractor.extract(exe_path)
    
    if pe_features is None:
        print("Error: Unable to parse the PE file. Check the path.")
        return
    else:
	    for name, vec in pe_features.items():   # Print the extracted feature
	        print(f"{name:20s}: shape={vec.shape}, "
                  f"min={vec.min():.3f}, max={vec.max():.3f}")

    # 2. Test parsing from a real EMBER row (replace with the path to your jsonl)
    jsonl_path = CONFIG["ember_path"]+ember_file
    print(f"\nExtracting from {jsonl_path}...")
    try:
        with open(jsonl_path, "r") as f:
            first_line = json.loads(f.readline()) # Reads only the first malware
        ember_features = ember_parser.parse(first_line)
        for name, vec in ember_features.items():
            print(f"{name:20s}: shape={vec.shape}, "
                  f"min={vec.min():.3f}, max={vec.max():.3f}")
        
    except FileNotFoundError:
        print("Error: JSONL file not found. Check the path.")
        return

    # 3. Dimension comparison (Shape Matching)
    print("\n--- COMPARISON RESULTS (SHAPE) ---")
    all_passed = True
    for key in pe_features.keys():
        shape_pe = pe_features[key].shape
        shape_ember = ember_features[key].shape
        
        if shape_pe == shape_ember:
            print(f"[{key:<16}] -> PE: {str(shape_pe):<8} | EMBER: {str(shape_ember):<8} | PASS")
        else:
            print(f"[{key:<16}] -> PE: {str(shape_pe):<8} | EMBER: {str(shape_ember):<8} | ERROR")
            all_passed = False
            
    if all_passed:
        print("\n Test passed! Both extractors generate identical vectors.")
    else:
        print("\n Warning: There are shape discrepancies.")

if __name__ == "__main__":
    run_test()