# DeepGraphDefense

**Graph Neural Networks for host-based malware detection and network intrusion detection, hardened with adversarial training.**

DeepGraphDefense is the code for a Master's thesis. It looks at how Graph Neural Networks (GNNs) can secure two levels of the same infrastructure:

- **Host level.** It classifies Windows Portable Executable (PE) files as benign or malicious.
- **Network level.** It detects malicious traffic in windows of network flows (NIDS).

Both detectors are trained with Generative Adversarial Networks (GANs) to make them harder to evade. They are compared against classical machine learning baselines, tested under concept drift and adversarial perturbations, and then combined into a single defense-in-depth pipeline.

---

## Table of Contents

- [Overview](#overview)
  - [Motivation](#motivation)
  - [Research Objectives](#research-objectives)
  - [Key Contributions](#key-contributions)
- [Architecture](#architecture)
  - [Host Domain: Malware Feature Graph (MFGraph)](#host-domain-malware-feature-graph-mfgraph)
  - [Network Domain: Flow Graph NIDS](#network-domain-flow-graph-nids)
  - [GNN Model: DGCNN with SortPooling](#gnn-model-dgcnn-with-sortpooling)
  - [Adversarial Augmentation: Dropout-WGAN-GP](#adversarial-augmentation-dropout-wgan-gp)
  - [Unified Defense-in-Depth Pipeline](#unified-defense-in-depth-pipeline)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Datasets](#datasets)
  - [EMBER 2018](#ember-2018)
  - [CIC-IDS2017](#cic-ids2017)
- [Usage](#usage)
  - [Part 1: Host-Based Malware Detection (PE)](#part-1-host-based-malware-detection-pe)
  - [Part 2: Network Intrusion Detection (NIDS)](#part-2-network-intrusion-detection-nids)
  - [Part 3: Evaluation and Unified Pipeline](#part-3-evaluation-and-unified-pipeline)
- [Configuration](#configuration)
- [Evaluation Methodology](#evaluation-methodology)
- [Performance](#performance)
- [Hardware and Training Time](#hardware-and-training-time)
  - [Training Environment](#training-environment)
  - [Execution Time](#execution-time)
- [Limitations and Known Issues](#limitations-and-known-issues)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

### Motivation

Most malware and intrusion detectors in use today are trained on flat feature vectors. This works on data that looks like the training set, but the models have two known weaknesses:

1. **Concept drift.** Malware families and attack patterns change over time, and accuracy drops on samples that appear after the training period.
2. **Adversarial evasion.** Small, targeted changes to a sample's features (padding bytes, altering packet timings) can push it past the decision boundary.

A graph representation keeps the *relationships* between features: how the parts of an executable relate to each other, or which network flows share a host or port. A model that learns from this structure, rather than from isolated values, can generalize better and is harder to fool with changes to single features.

### Research Objectives

- Build graph representations of PE executables and of network traffic.
- Train a DGCNN-based classifier for each domain.
- Use a GAN to generate synthetic malicious samples and add them to the training data to improve robustness.
- Compare the GNNs with classical baselines (Logistic Regression, Decision Tree, SVM, MLP, FFDNN).
- Measure how performance holds up over time (concept drift) and under adversarial perturbation.
- Measure the value of combining network and host detection in one pipeline.

### Key Contributions

- **MFGraph:** a 9-node graph for each PE file. It is built either from raw executables (with LIEF) or from EMBER 2018 feature vectors.
- **Flow graphs:** CIC-IDS2017 flows grouped into fixed-size windows. Edges link flows that share a source IP, a source–destination pair or a destination port, or that are adjacent in time.
- **Dropout-WGAN-GP:** a graph-aware generator trained against several critics. Its synthetic malicious graphs are added to the training set.
- **Evaluation framework:** concept drift, adversarial robustness, baseline comparison and an *impact mitigation* metric for the unified pipeline.

---

## Architecture

```
                 ┌──────────────────────── Network layer (traffic analysis) ───────────────────────┐
 CIC-IDS2017 ───►│ CSV flows ─► window (N flows) ─► Flow Graph ─► DGCNN (NIDS) ─► p_nids ─► calib. │──► r_nids ─┐
                 └─────────────────────────────────────────────────────────────────────────────────┘            │
                 ┌────────────────────────── Host layer (file analysis) ───────────────────────────┐            ▼
 EMBER / .exe ──►│ PE file ─► 9 feature groups ─► MFGraph ─► DGCNN / DGCNN+GAN ─► p_pe ─► calib.   │──► r_pe ──► Unified score
                 └─────────────────────────────────────────────────────────────────────────────────┘          ─► alert + severity
                            ▲                                                                ▲                ─► Impact Mitigation
                            └── Dropout-WGAN-GP synthetic malicious graphs (training only) ──┘
```

Both detectors share the same graph classifier. Their outputs are calibrated to a common risk scale and then fused into a single **unified alert score** for each monitored host session.

### Host Domain: Malware Feature Graph (MFGraph)

Each executable becomes a graph with **9 nodes**, one per EMBER feature group:

| Node | Feature group      | Node | Feature group       |
|------|--------------------|------|---------------------|
| 0    | `general`          | 5    | `byte_histogram`    |
| 1    | `header`           | 6    | `byte_entropy`      |
| 2    | `imported`         | 7    | `data_directories`  |
| 3    | `exported`         | 8    | `string`            |
| 4    | `section`          |      |                     |

The edges are fixed and undirected. `general` connects to `imported` and `exported`, and `string` connects to every structural and statistical node. Node feature vectors are zero-padded to the same length (`max_feature_len`). The features come from two sources:

- `PEFeatureExtractor`: parses raw `.exe` files with LIEF.
- `EmberFeatureParser`: reads the same 9 groups from the EMBER 2018 `.jsonl` files.

### Network Domain: Flow Graph NIDS

The CIC-IDS2017 CSVs are cut into windows of consecutive flows. Each flow is a node. Its features are flow statistics (packet lengths and counts, inter-arrival times, byte and packet rates, TCP flags, duration), log-scaled with `log1p` to compress extreme values. Edges are added between flows that:

- come from the same **source IP**,
- share the same **source–destination IP pair**,
- target the same **destination port**, or
- are **next to each other in time**.

Large groups are linked as a chain rather than all-to-all, so the graph does not become too dense. A window is labelled **malicious when more than 30% of its flows are attacks**, so a few stray attack flows do not mark the whole window. During training, the class imbalance is handled with a `pos_weight` computed from the benign/malicious ratio of the training set.

### GNN Model: DGCNN with SortPooling

`MFGraph` ([src/models/gnn.py](src/models/gnn.py)) implements a Deep Graph Convolutional Neural Network:

1. Three stacked `GCNConv` layers. The outputs of all three layers are concatenated.
2. **SortPooling** keeps the top-`k` nodes, which gives a fixed-size representation.
3. A learned linear readout over the `k` nodes.
4. A three-layer MLP classifier with dropout that outputs one logit (benign or malicious).

### Adversarial Augmentation: Dropout-WGAN-GP

[src/models/gan.py](src/models/gan.py) and [src/training/gan_pipeline.py](src/training/gan_pipeline.py) define the GAN, which is shared by both domains:

- **Generator:** a graph-aware generator that produces node features *on top of real malicious graph structures*. It combines graph-level and node-level noise with a positional embedding of each node (its feature group for PE, its position in the window for NIDS), then refines them with two residual `GCNConv` layers. A sigmoid maps the outputs into the real feature range of the malicious class, so synthetic samples stay in the same space as real ones.
- **Critics:** several DGCNN-based critics with no sigmoid, trained with a Wasserstein loss and a gradient penalty (WGAN-GP, `n_critic` critic steps per generator step). Following Dropout-GAN, each critic is randomly dropped at each step, so the generator cannot overfit to a single critic.
- **Augmentation:** the trained generator creates synthetic malicious graphs (`synthetic_ratio` × real malicious count). The final classifier is trained on the real data plus these synthetic graphs.

### Unified Defense-in-Depth Pipeline

[src/evaluation/unified_pipeline.py](src/evaluation/unified_pipeline.py) simulates a monitored environment and combines the two detectors into a single decision for each session.

**1. Session simulation.** The pipeline generates host sessions (2,000 by default). Each session is a time-ordered stream of network windows and file events, drawn from the test pools:

| Scenario         | Share | Malicious traffic | Malware on host |
|------------------|-------|-------------------|-----------------|
| `benign`         | 70%   | –                 | –               |
| `network_attack` | 10%   | ✓                 | –               |
| `malware_drop`   | 10%   | –                 | ✓               |
| `multi_stage`    | 10%   | ✓                 | ✓               |

In `multi_stage` sessions, the order follows the kill chain: malicious traffic (reconnaissance or exploitation) always comes before the malware reaches the host.

**2. Risk calibration.** Raw probabilities from two different models cannot be compared directly. For each detector, the best operating threshold *t* is chosen on held-out data with Youden's J statistic (max TPR − FPR). The probability is then mapped piecewise-linearly so that *t* falls exactly at 0.5:

$$
r(p) =
\begin{cases}
0.5 \cdot \dfrac{p}{t} & p < t \\[2mm]
0.5 + 0.5 \cdot \dfrac{p - t}{1 - t} & p \ge t
\end{cases}
$$

**3. Unified score.** Each domain contributes its strongest event in the session, $R_{nids} = w_{nids} \cdot \max r_{nids}$ and $R_{pe} = w_{pe} \cdot \max r_{pe}$. These are fused as follows:

$$
S =
\begin{cases}
1 - (1 - R_{nids})(1 - R_{pe}) & \text{if } R_{nids} \ge \tau \text{ and } R_{pe} \ge \tau \quad \text{(corroborated: Noisy-OR boost)} \\[1mm]
\max(R_{nids},\, R_{pe}) & \text{otherwise}
\end{cases}
$$

with alert threshold $\tau = 0.5$. When both layers agree, the score is boosted towards 1. When only one fires, its signal is kept unchanged. The score rises with both risks and crosses $\tau$ as soon as either detector fires. Only the maximum event per domain is used, because a Noisy-OR over every event would push long benign sessions towards 1.

The unified score sets the severity of the alert:

| Unified score | Alert level |
|---------------|-------------|
| ≥ 0.9         | `CRITICAL`  |
| ≥ 0.7         | `HIGH`      |
| ≥ 0.5         | `MEDIUM`    |
| ≥ 0.3         | `LOW`       |
| < 0.3         | `NONE`      |

**4. Impact Mitigation.** For each threat session, the pipeline goes through the events in time order. The first malicious event that is correctly detected sets the outcome:

| Outcome                                              | Mitigation weight |
|------------------------------------------------------|-------------------|
| Stopped at the network layer (before delivery)       | 1.0               |
| Stopped at the host layer (malware reached the host) | 0.5               |
| Never detected                                       | 0.0               |

The **Impact Mitigation (IM)** score is the mean weight over threat sessions. The **net IM** also subtracts a cost of 0.1 for each false alert on a benign session, so a detector that flags everything gets no credit:

$$
IM_{net} = \frac{\sum_{\text{threats}} w_{\text{outcome}} \;-\; 0.1 \cdot \#\text{false alerts}}{\#\text{threat sessions}}
$$

The pipeline reports event-level metrics (per detector), session-level metrics (on the unified score), IM, net IM and the false-alarm rate. It can be run with the standard PE GNN or with the GAN-hardened one (`pe_model: "gnn" | "gan"`).

---

## Project Structure

```text
thesis-project/
├── orchestrator.py                 # single CLI entry point for every pipeline task
│
├── data/
│   ├── raw/                        # EMBER 2018 (.jsonl), CIC-IDS2017 (.csv), PE samples
│   ├── processed/                  # intermediate extracted features
│   └── graphs/                     # serialized PyG graphs (.pt)
│       ├── ember_graphs/{train,test/MM}
│       └── nids_graphs/
│
├── src/
│   ├── features/
│   │   └── pe_extractor.py         # LIEF PE extractor + EMBER feature parser (9 groups)
│   ├── graph/
│   │   ├── feature_graph.py        # MFGraph builder (9-node PE topology)
│   │   ├── ember_graph.py          # EMBER → graphs, split by month for drift evaluation
│   │   └── cic_graph.py            # CIC-IDS2017 → windowed flow graphs
│   ├── models/
│   │   ├── gnn.py                  # DGCNN + SortPooling + MLP classifier
│   │   └── gan.py                  # graph generator + DGCNN critics
│   ├── training/
│   │   ├── pe_dataset.py           # PE graph dataset loader
│   │   ├── nids_dataset.py         # NIDS graph dataset and splits
│   │   ├── train_gnn_pe.py         # GNN training: PE domain
│   │   ├── train_gnn_nids.py       # GNN training: NIDS domain
│   │   ├── gan_pipeline.py         # shared Dropout-WGAN-GP + augmented classifier pipeline
│   │   ├── train_gan_pe.py         # GAN pipeline: PE domain
│   │   └── train_gan_nids.py       # GAN pipeline: NIDS domain
│   ├── evaluation/
│   │   ├── concept_drift.py        # performance over time (EMBER months / NIDS chunks)
│   │   ├── pe_adversarial.py       # GAN-driven perturbation attacks on PE detectors
│   │   ├── nids_adversarial.py     # feature-scaling attacks (timing / packet size) on NIDS
│   │   ├── plot_metrics.py         # comparison tables and plots
│   │   └── unified_pipeline.py     # NIDS + PE defense-in-depth simulation
│   └── utils/
│       ├── config.py               # paths and hyperparameters
│       └── logger.py               # experiment logging
│
├── baselines/
│   ├── train_baseline_pe.py        # classical ML baselines on flattened PE features
│   └── train_baseline_nids.py      # classical ML baselines on flattened flow features
│
├── notebooks/                      # exploration, graph visualization, result analysis
├── experiments/                    # checkpoints, logs, metrics and plots
├── tests/                          # unit tests (extractor, graph builder, logger)
└── envs/                           # conda environments
```

---

## Prerequisites

- Linux or macOS (tested on Linux)
- [Conda](https://docs.conda.io/) or Mamba
- Python 3.10
- PyTorch and PyTorch Geometric. The environment installs the CPU build by default; a CUDA GPU is optional but recommended for GAN training however it needs to implement.
- About 50 GB of free disk space for the raw datasets and the generated graphs
- At least 8 or 16 GB of RAM for the model training

---

## Installation

```bash
git clone <repository-url> thesis-project
cd thesis-project

# Main environment (PyTorch, PyG, LIEF, scikit-learn, ...)
conda env create -f envs/env_thesis_proj.yml
conda activate thesis-gnn

# Optional: legacy environment for the original EMBER feature extraction (Python 3.7, LightGBM)
conda env create -f envs/env_ember.yml
```

For GPU support, replace the CPU wheel index in `envs/env_thesis_proj.yml` with the CUDA build that matches your system.

---

## Datasets

### EMBER 2018

[EMBER](https://github.com/elastic/ember) (Endgame Malware BEnchmark for Research) contains static features extracted from about 1 million Windows PE files, labelled benign, malicious or unlabelled.

- Download the 2018 feature release and extract the `.jsonl` files to `data/raw/ember2018/`.
- Unlabelled samples are skipped.
- Samples are **split by the `appeared` month**: January 2018 is used for training, and February–December 2018 are kept as separate monthly test sets (`test/02` … `test/12`) to measure concept drift.

### CIC-IDS2017

[CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) (Canadian Institute for Cybersecurity) contains five days of labelled network traffic, including DoS/DDoS, brute force, web attacks, infiltration, botnet and port scan.

- Place the `MachineLearningCSV` / `TrafficLabelling` CSV files in `data/raw/CIC-IDS-2017/`.
- Rows containing `NaN` or `Inf` are removed, and every non-`BENIGN` label counts as malicious.

---

## Usage

Every task runs through the orchestrator:

```bash
python orchestrator.py --task <TASK>
```

### Part 1: Host-Based Malware Detection (PE)

| Task                  | Description                                                              |
|-----------------------|--------------------------------------------------------------------------|
| `ember_graph`         | Build MFGraphs from the EMBER `.jsonl` files, with monthly train/test split |
| `extract_pe`          | Extract features from raw `.exe` files with LIEF and build their graphs   |
| `train_pe_gnn`           | Train the DGCNN PE classifier (builds the graphs first if they are missing) |
| `train_pe_gan`           | Train the Dropout-WGAN-GP and the GAN-augmented PE classifier             |
| `train_pe_baselines`  | Train the classical ML baselines on flattened PE features                 |

```bash
python orchestrator.py -t ember_graph
python orchestrator.py -t train_pe_gnn
python orchestrator.py -t train_pe_gan
python orchestrator.py -t train_pe_baselines
```

### Part 2: Network Intrusion Detection (NIDS)

| Task                    | Description                                                    |
|-------------------------|----------------------------------------------------------------|
| `build_cicids`          | Turn the CIC-IDS2017 CSV files into windowed flow graphs       |
| `train_nids_gnn`            | Train the DGCNN NIDS classifier                                |
| `train_nids_gan`        | Train the Dropout-WGAN-GP and the GAN-augmented NIDS classifier |
| `train_nids_baselines`  | Train the classical ML baselines on flattened flow features     |

```bash
python orchestrator.py -t build_cicids
python orchestrator.py -t train_nids_gnn
python orchestrator.py -t train_nids_gan
python orchestrator.py -t train_nids_baselines
```

### Part 3: Evaluation and Unified Pipeline

| Task                     | Description                                                      |
|--------------------------|------------------------------------------------------------------|
| `eval_drift`             | Concept drift evaluation (EMBER months, NIDS chunks)             |
| `eval_pe_adversarial`    | Robustness against GAN-generated perturbations of PE features     |
| `eval_nids_adversarial`  | Robustness against scaled timing and packet-size features         |
| `plot_metrics`           | Comparison tables (CSV) and plots (PNG) across all models         |
| `unified_pipeline`       | Simulate NIDS + PE defense-in-depth and compute Impact Mitigation |

```bash
python orchestrator.py -t eval_drift
python orchestrator.py -t eval_pe_adversarial
python orchestrator.py -t eval_nids_adversarial
python orchestrator.py -t plot_metrics
python orchestrator.py -t unified_pipeline
```

All outputs (checkpoints, logs, metrics and figures) are written to `experiments/` in timestamped subfolders.

---

## Configuration

All paths and hyperparameters are set in [src/utils/config.py](src/utils/config.py):

| Section | Key parameters |
|---------|----------------|
| Paths   | `ember_path`, `ember_graphs_path`, `cic_graphs_path`, `experiments_path` |
| Graph   | `feature_names`, per-group dimensions, `max_feature_len` |
| `gnn`   | `batch_size=256`, `learning_rate=2e-3`, `num_epochs=20`, `hidden_dim=48`, `num_layers=3`, `k=28`, `dropout_rate=0.5` |
| `gan`   | `epochs=20`, `noise_dim=64`, `num_discriminators=3`, `dropout_threshold=0.5`, `n_critic=5`, `gp_lambda=10`, `synthetic_ratio=0.5` |
| `nids`  | `input_dim`, `learning_rate` |

---

## Evaluation Methodology

- **Metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC, and Impact Mitigation for the unified pipeline.
- **Baselines:** Logistic Regression, Decision Tree, SVM, MLP and FFDNN, trained on the same features flattened into vectors.
- **Concept drift:** train on the earliest period, then test on each later period separately.
- **Adversarial robustness:** apply perturbations of increasing strength to malicious samples and measure the drop in detection rate.

---

## Performance

> Results are produced by `plot_metrics` and stored in `experiments/model_comparisons/<timestamp>/`. The tables below will be filled in with the final thesis results.

| Domain | Model            | Accuracy | Precision | Recall | F1 | ROC-AUC |
|--------|------------------|----------|-----------|--------|----|---------|
| PE     | DGCNN            |          |           |        |    |         |
| PE     | DGCNN + GAN      |          |           |        |    |         |
| PE     | Best baseline    |          |           |        |    |         |
| NIDS   | DGCNN            |          |           |        |    |         |
| NIDS   | DGCNN + GAN      |          |           |        |    |         |
| NIDS   | Best baseline    |          |           |        |    |         |

---

## Hardware and Training Time

### Training Environment

All models were trained and evaluated on the following machine:

| Component | Specification |
|-----------|---------------|
| CPU       | Intel Core i7-6700 @ 3.40 GHz (4 cores / 8 threads) |
| GPU       | NVIDIA GeForce GTX 950, 2 GB VRAM (driver 580.178.04) |
| RAM       | 16 GB |
| Storage   | 1 TB SSD/HDD |
| OS        | EndeavourOS (Arch Linux), kernel 6.18 LTS |
| Python    | 3.10 (conda environment `thesis-gnn`) |
| Device used for training | CPU / CUDA |

### Execution Time

Wall-clock time for a single run of each task, using the default configuration in [src/utils/config.py](src/utils/config.py).

| Task (`orchestrator.py -t`) | Model / Step                         | Execution time |
|-----------------------------|--------------------------------------|----------------|
| `ember_graph`               | EMBER → MFGraph construction         |    33m 30s     |
| `build_cicids`              | CIC-IDS2017 → flow graph construction |     1m 12s     |
| `train_pe_gnn`              | GNN PE (DGCNN)                       |    17m 57s    |
| `train_pe_gan`              | GAN PE (Dropout-WGAN-GP + classifier)|   3h 11m 40s   |
| `train_pe_baselines`        | PE baselines (LR, DT, SVM, MLP, FFDNN)|    14m 14s    |
| `train_nids_gnn`            | GNN NIDS (DGCNN)                     |     3m 21s     |
| `train_nids_gan`            | GAN NIDS (Dropout-WGAN-GP + classifier) |     10m 46s    |
| `train_nids_baselines`      | NIDS baselines (LR, DT, SVM, MLP, FFDNN) |     21s        |
| `eval_drift`                | Concept drift evaluation             |   2h 47m 47s   |
| `eval_pe_adversarial`       | PE adversarial evaluation            |     1m 18s     |
| `eval_nids_adversarial`     | NIDS adversarial evaluation          |      13s       |
| `unified_pipeline`          | Unified defense-in-depth simulation  |     2m 50s     |

---

## Limitations and Known Issues

- **Static analysis only.** PE features are static, so packed or obfuscated binaries may hide their behavior.
- **Fixed PE topology.** The 9-node MFGraph structure is defined by hand, not learned.
- **Window-level NIDS labels.** Flow graphs are labelled per window, so attacks that cover only a few flows in a window may be underrepresented.
- **Dataset age.** EMBER 2018 and CIC-IDS2017 are well-known benchmarks but do not include recent threats.
- **Feature-space attacks.** The adversarial evaluation changes feature vectors, not real binaries or packets, so the perturbed samples may not correspond to working malware or traffic.
- **Memory.** Baseline training uses a subset of the PE samples (10,000) to avoid running out of RAM.

---

## Testing

```bash
pytest tests/
```

The unit tests cover the PE feature extractor, the MFGraph builder and the logger.

---

## Contributing

This repository belongs to an ongoing Master's thesis. Issues and suggestions are welcome. Please open an issue before submitting a pull request, and follow the existing code style.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
