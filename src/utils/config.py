"""
Single source of truth for paths and hyperparameters.
Every training run dumps this dict to experiments/<task>/<run>/config.json,
so a result can always be traced back to the configuration that produced it.
"""
CONFIG = {
    # Paths (relative to the project root)
    "ember_path": "data/raw/ember2018",
    "PEsamples_path": "data/raw/test_samples",
    "processed_path": "data/processed",
    "ember_graphs_path": "data/graphs/ember_graphs",
    "cic_graphs_path": "data/graphs/nids_graphs",
    "experiments_path": "experiments",

    # Unused by the current pipeline (kept for older experiments)
    "num_features": 100,

    # PE graph: one node per feature family, in this order (node index = list index)
    "feature_names": [
        "general", "header", "imported", "exported",
        "section", "byte_histogram", "byte_entropy",
        "data_directories", "string"
    ],

    # Raw length of each PE node vector before padding to max_feature_len
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

    # Reference split ratios. The training scripts hard-code the same values:
    # PE = 80/20 random split of January 2018, NIDS = 80/10/10 chronological per scenario.
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
    "random_seed": 42,

    # --- MODEL SPECIFIC CONFIGURATIONS ---

    "gnn": {
        "batch_size": 256,
        "learning_rate": 0.002,
        "num_epochs": 20,
        "hidden_dim": 48,
        "num_layers": 3,            # must match the 3 GCNConv layers hard-coded in MFGraph
        "dropout_rate": 0.5,
        "mlp_hidden_dim": 1024,
        "k": 28                     # SortPooling: nodes kept per graph (PE graphs have 9 -> zero-padded)
    },

    "gan": {
        "epochs": 20,
        "batch_size": 64,
        "noise_dim": 64,
        "dropout_threshold": 0.5,   # probability of dropping each critic (Dropout-GAN)
        "num_discriminators": 3,
        "n_critic": 5,              # critic steps per generator step (WGAN-GP)
        "gp_lambda": 10.0,
        "lr_generator": 1e-4,
        "lr_discriminator": 1e-4,
        "synthetic_ratio": 0.5      # synthetic malicious graphs, relative to real malicious count
    },

    # NIDS-specific overrides (the GNN architecture is shared via "gnn")
    "nids": {
        "input_dim": 16,            # flow features per node, see FEATURE_COLS in src/graph/cic_graph.py
        "learning_rate": 0.002
    },

    # PE node vectors are zero-padded / truncated to this length (= PE GNN input_dim)
    "max_feature_len": 512,

    # Optional overrides for src/evaluation/unified_pipeline.py (see UNIFIED_DEFAULTS there), e.g.
    # "unified": {"pe_model": "gan", "num_sessions": 5000},

}
