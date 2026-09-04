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
│   ├── processed/            # extracted feature
│   ├── graphs/               # builded graphs (.pt o .json)
│   └── test_samples          # .exe files for test the extraction of the features
│   
│
├── src/
│   ├── features/
│   │   ├── pe_extractor.py   # LIEF → 2 Classes: 9 static features (MFGraph) from new and same 9 features from ember dataset
│   │   └── flow_extractor.py # pcap/csv → flow features (GNN-NIDS)
│   │
│   ├── graph/
│   │   ├── feature_graph.py  # build the feature graph (MFGraph)
│   │   └── host_graph.py     # build host-connection graph (NIDS)
│   │
│   ├── models/
│   │   ├── gnn.py            # DGCNN + readout
│   │   ├── gan.py            # Generator + Discriminator (Dropout-GAN)
│   │   └── classifier.py     # final MLP
│   │
│   ├── training/
│   │   ├── train_gan.py      # phase 1: train the GAN
│   │   ├── train_gnn.py      # phase 2: train the GNN
│   │   └── evaluate.py       # metrics + concept drift eval
│   │
│   └── utils/
│       ├── metrics.py        # AUC, F1, impact mitigation
│       ├── logger.py         # tracking logs (wandb/csv)
│       └── config.py         # hyperparameters
│
├── baselines                 # comparison models
│   ├── mlp/
│   │   ├── dataset.py        # logic to load flat vectors (without building graphs)
│   │   ├── model.py          # standard MLP neural network (Dense layers only)
│   │   └── train_mlp.py      # specific training script for the MLP
│   │
│   ├── lightgbm/             # gradient Boosting (Lower bound)
│   │   └── train_lgbm.py     # specific training script for the lightgbm
│   │
│   └── random_forest/        # traditional tree model
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_graph_visualization.ipynb
│   └── 03_results_analysis.ipynb
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