# DeepGraphDefense - GNN
Graph Neural Networks and Other Machine Learning Techniques for Malware Analysis and the Implementation of a NIDS

## Table of Contents

## Overview

## Architecture

## Project structure
thesis-project/
├── data/
│   ├── raw/                  # dataset
│   ├── processed/            # extracted feature
│   └── graphs/               # builded graphs (.pt o .json)
│
├── src/
│   ├── features/
│   │   ├── pe_extractor.py   # LIEF → 9 static feature (MFGraph)
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
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_graph_visualization.ipynb
│   └── 03_results_analysis.ipynb
│
├── experiments/              # output: saved models, log, plot
└── tests/                    # modules unit test