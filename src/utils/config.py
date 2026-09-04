CONFIG = {
    # Paths
    "ember_path": "data/raw/ember2018",
    "PEsamples_path": "data/raw/test_samples",
    "processed_path": "data/processed",
    "graphs_path": "data/graphs",
    "experiments_path": "experiments",

    # Feature selection
    "num_features": 100,

    # Graph construction
    "feature_names": [
        "general", "header", "imported", "exported",
        "section", "byte_histogram", "byte_entropy",
        "data_directories", "string"
    ],

    # Feature dimensions (fixed vector size per node)
    "general_dim":    10,
    "header_dim":     10,
    "imported_dim":   10,   # max DLLs considered
    "exported_dim":   1,
    "section_max":    5,    # max sections considered
    "section_dim":    15,   # section_max * 3 values
    "entropy_dim":    256,
    "datadirs_max":   16,
    "datadirs_dim":   32,   # datadirs_max * 2 values
    "string_dim":     6,
    "byte_histogram_dim": 256,

    # Global Training (Default)
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
    "random_seed": 42,

    # --- MODEL SPECIFIC CONFIGURATIONS ---

    # GNN Parameters
    "gnn": {
        "batch_size": 256,
        "learning_rate": 0.002,
        "num_epochs": 50,
        "hidden_dim": 128,
        "num_layers": 3,
        "dropout_rate": 0.5,
        "num_classes": 2,
    },

    # MLP Baseline Parameters
    "mlp": {
        "input_dim": 2381,       # EMBER flat feature size
        "batch_size": 2048,      # Large batch for fast tabular training
        "learning_rate": 0.001,
        "num_epochs": 20,        # MLPs converge much faster than GNNs
        "hidden_dims": [1024, 512, 256],
        "dropout_rate": 0.3,
    },

    # LightGBM Baseline Parameters
    "lgbm": {
        "n_estimators": 1000,    # Number of trees (EMBER paper uses 1000-2000)
        "learning_rate": 0.05,
        "num_leaves": 31,
        "objective": "binary",
    }

    }
