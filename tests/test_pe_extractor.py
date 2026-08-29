import sys
sys.path.append("/home/simone/UNI/Master/Tesi/thesis-project")

from src.features.pe_extractor import PEFeatureExtractor

extractor = PEFeatureExtractor()
features = extractor.extract("data/raw/test_samples/putty.exe")

if features:
    for name, vec in features.items():
        print(f"{name:20s}: shape={vec.shape}, "
              f"min={vec.min():.3f}, max={vec.max():.3f}")
else:
    print("Estrazione fallita")