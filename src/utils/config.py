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

    # Training
    "batch_size": 256,
    "learning_rate": 0.002,
    "num_epochs": 50,
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
    "random_seed": 42,

    # GNN
    "hidden_dim": 128,
    "num_layers": 3,
    "dropout_rate": 0.5,
    "num_classes": 2,
    }
