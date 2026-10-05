# DeepGraphDefense - GNN
Graph Neural Networks and Other Machine Learning Techniques for Malware Analysis and the Implementation of a NIDS

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Part 1: ]
- [Part 2: ]
- [Dataset](#dataset)
- [Performance](#performance)
- [Limitations and Known Issues](#limitations-and-known-issues)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## Overview

## Architecture

## Project structure
```
thesis-project/
├── orchestrator.py           # entrypoint for all the functionality
├── data/
│   ├── raw/                  # dataset
│   ├── processed/            # extracted feature - TODO
│   └── graphs/               # builded graphs (.pt o .json)
│   
│
├── src/
│   ├── features/
│   │   ├── pe_extractor.py   # LIEF → 2 Classes: 9 static features (MFGraph) from new and same 9 features from ember dataset
│   │   └── flow_extractor.py # pcap/csv → flow features (GNN-NIDS) - TODO
│   │
│   ├── graph/
│   │   ├── feature_graph.py  # build the feature graph from new exe samples(MFGraph)
│   │   ├── ember_graph.py    # build the feature graph from ember dataset to train the model
│   │   └── host_graph.py     # build host-connection graph (NIDS) - TODO
│   │
│   ├── models/
│   │   ├── gnn.py            # DGCNN + readout
│   │   ├── gan.py            # Generator + Discriminator (Dropout-GAN)
│   │   └── classifier.py     # final MLP - TODO
│   │
│   ├── training/
│   │   ├── train_gan.py      # phase 1: train the GAN
│   │   ├── train_gnn.py      # phase 2: train the GNN
│   │   ├── pe_dataset.py     # building dataset to generate .pt files from pe ember samples
│   │   ├── nids_dataset.py   # building dataset to generate .pt files from csv CIC dataset
│   │   └── train_nids.py   
│   │
│   ├── evaluation
│   │   ├── concept_drift.py  #   
│   │   ├── plot_metrics.py   # visualitation fof AUC and loss from all the baseline models
│   │   ├── nids_adversial.py # loads the trained GNN and systematically applies a multiplier to specific feature indices to simulate an attacker slowing down packets or manipulating packet sizes
│   │   └── unified_pipeline.py 
│   │ 
│   └── utils/
│       ├── metrics.py        # AUC, F1, impact mitigation - TODO
│       ├── logger.py         # tracking logs (wandb/csv)
│       └── config.py         # hyperparameters
│
├── baselines                 # comparison models
│   ├── mlp/
│   │   ├── model.py          # standard MLP neural network (Dense layers only)
│   │   └── train_mlp.py      # specific training script for the MLP
│   │
│   ├── lightgbm/             # gradient Boosting (Lower bound)
│   │   └── train_lgbm.py     # specific training script for the lightgbm
│   │
│   └── random_forest/        # traditional tree model - TODO
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_graph_visualization.ipynb
│   ├── 03_results_analysis.ipynb
│   ├── 04_adversarial_robustness.ipynb
│   └── 05_unified_pipeline_impact.ipynb
│
├── experiments/              # output: saved models, log, plot
├── tests/                    # modules unit test
└── envs/                     # environment for the requirements installation
```

## Dataset
### Ember

### CIC-IDS2017


## License
This project is licensed under the MIT License. See `LICENSE` file for details.