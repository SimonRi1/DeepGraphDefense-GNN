CONFIG = {
    # Paths
    "ember_path": "data/raw/ember2018",
    "processed_path": "data/processed",
    "graphs_path": "data/graphs",
    "experiments_path": "experiments",

    # Feature selection
    "num_features": 100,          # top N feature per information gain

    # Graph construction
    "feature_names": [            # le 9 feature di MFGraph
        "general", "header", "imported", "exported",
        "section", "byte_histogram", "byte_entropy",
        "data_directories", "string"
    ],

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
    "num_classes": 2,             # benigno / maligno
    }
