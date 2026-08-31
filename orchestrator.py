import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.features.pe_extractor import PEFeatureExtractor
from src.graph.feature_graph import FeatureGraphBuilder
from src.utils.logger import ExperimentLogger
# from src.models.gnn import MyGraphNeuralNetwork (You will build this later)

def main():
    # 1. Setup Logger
    logger = ExperimentLogger(experiment_name="GNN_Training", config={"epochs": 100})
    
    # 2. Extract Data
    extractor = PEFeatureExtractor()
    builder = FeatureGraphBuilder()
    
    # 3. Train Model
    # model = MyGraphNeuralNetwork()
    # for epoch in range(100):
    #     loss = model.train_step()
    #     logger.log_epoch(epoch, {"loss": loss})

if __name__ == "__main__":
    main()