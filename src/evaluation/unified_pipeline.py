"""
Unified Detection Pipeline (File Analysis + Traffic Analysis).

Simulates a monitored environment made of host sessions. Every session is a
chronological stream of events: network traffic windows (NIDS graphs) and
software downloaded/executed on the host (PE graphs). Each event receives an
individual verdict, and every session receives an aggregated alert score in
[0, 1] with a severity level.

Steps:
    1. Score the NIDS and PE test pools once with the trained MFGraph models.
    2. Calibrate each detector on held-out data (threshold -> risk 0.5).
    3. Generate mixed sessions (normal traffic, network attacks, malware
       drops, multi-stage intrusions) with a realistic benign majority.
    4. Fuse per-domain max risks into a unified score (Noisy-OR boost when
       both detectors agree, strongest signal otherwise) + alert level.
    5. Compute event-level, session-level and Impact Mitigation metrics.
"""
import sys
import json
import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from sklearn.metrics import (f1_score, roc_auc_score, roc_curve, precision_score,
                             recall_score, confusion_matrix)

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models.gnn import MFGraph
from src.training.pe_dataset import PEGraphDataset
from src.training.nids_dataset import get_nids_splits
from src.utils.config import CONFIG

UNIFIED_DEFAULTS = {
    "num_sessions": 2000,
    "scenario_mix": {               # fraction of sessions per scenario type
        "benign": 0.70,
        "network_attack": 0.10,
        "malware_drop": 0.10,
        "multi_stage": 0.10,
    },
    "net_events_per_session": (3, 8),
    "file_events_per_session": (1, 4),
    "pe_pool_per_class": 2000,      # PE test graphs scored per class
    "pe_calib_per_class": 1000,     # PE train graphs used for calibration
    "domain_weights": {"nids": 1.0, "pe": 1.0},
    "alert_threshold": 0.5,
    "alert_levels": [(0.9, "CRITICAL"), (0.7, "HIGH"), (0.5, "MEDIUM"), (0.3, "LOW"), (0.0, "NONE")],
    "mitigation_weights": {"network": 1.0, "host": 0.5, "missed": 0.0},
    "false_alarm_cost": 0.1,        # penalty per false alert, relative to a full mitigation
    "pe_model": "gnn",              # "gnn" (baseline) or "gan" (adversarially hardened)
    "seed": 42,
}


def get_unified_config():
    cfg = dict(UNIFIED_DEFAULTS)
    cfg.update(CONFIG.get("unified", {}))
    return cfg


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def find_latest_checkpoint(search_dir: Path, filename: str):
    """Finds the most recent checkpoint with the given file name."""
    checkpoints = list(search_dir.rglob(filename))
    if not checkpoints:
        raise FileNotFoundError(f"No '{filename}' found in {search_dir}. Train the model first.")
    return max(checkpoints, key=lambda f: f.stat().st_mtime)


def build_model(weights_path, input_dim, device):
    gnn_config = CONFIG["gnn"]
    model = MFGraph(
        input_dim=input_dim,
        hidden_dim=gnn_config["hidden_dim"],
        k=gnn_config["k"],
        dropout_rate=gnn_config.get("dropout_rate", 0.5)
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def load_unified_system(device, pe_input_dim, nids_input_dim, pe_model_kind="gnn"):
    """Loads the PE (host) and NIDS (network) detectors."""
    exp_base = project_root / CONFIG.get("experiments_path", "experiments")

    if pe_model_kind == "gan":
        pe_weights = find_latest_checkpoint(exp_base / "gan_training", "model_gan.pth")
    else:
        pe_weights = find_latest_checkpoint(exp_base / "gnn_training", "best_gnn_model.pth")
    nids_weights = find_latest_checkpoint(exp_base / "nids_gnn", "model_gnn.pth")

    print(f"[Unified Pipeline] PE weights:   {pe_weights}")
    print(f"[Unified Pipeline] NIDS weights: {nids_weights}")
    return build_model(pe_weights, pe_input_dim, device), build_model(nids_weights, nids_input_dim, device)


# ---------------------------------------------------------------------------
# Scoring & calibration
# ---------------------------------------------------------------------------
@torch.no_grad()
def score_graphs(model, graphs, device, desc, batch_size=64):
    """Returns (probabilities, labels) for a list/dataset of graphs."""
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    probs, labels = [], []
    for batch in tqdm(loader, desc=desc):
        batch = batch.to(device)
        # NIDS graphs are already scaled by the graph builder: no extra log1p.
        out = model(batch.x, batch.edge_index, batch.batch).view(-1)
        probs.append(torch.sigmoid(out).cpu().numpy())
        labels.append(batch.y.view(-1).cpu().numpy())
    return np.concatenate(probs), np.concatenate(labels).astype(int)


def sample_balanced(dataset, per_class, seed):
    """Lazily draws up to `per_class` graphs per label from a large dataset."""
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    pools = {0: [], 1: []}
    for idx in tqdm(indices, desc="Sampling PE graphs"):
        data = dataset[idx]
        label = int(data.y.item())
        if label in pools and len(pools[label]) < per_class:
            pools[label].append(data)
        if all(len(p) >= per_class for p in pools.values()):
            break
    return pools[0] + pools[1]


class ThresholdCalibrator:
    """
    Maps raw probabilities to a calibrated risk in [0, 1] such that the
    detector's optimal operating threshold (Youden's J on held-out data)
    lands exactly at 0.5. This makes the two detectors comparable before
    they are fused into the unified score.
    """
    def __init__(self, name):
        self.name = name
        self.threshold = 0.5

    def fit(self, probs, labels):
        if len(np.unique(labels)) < 2:
            print(f"[Calibration] {self.name}: single-class calibration set, keeping threshold 0.5")
            return self
        fpr, tpr, thr = roc_curve(labels, probs)
        best = np.argmax(tpr - fpr)
        self.threshold = float(np.clip(thr[best], 1e-6, 1 - 1e-6))
        print(f"[Calibration] {self.name}: operating threshold = {self.threshold:.4f}")
        return self

    def transform(self, probs):
        probs = np.asarray(probs, dtype=float)
        t = self.threshold
        return np.where(probs < t, 0.5 * probs / t, 0.5 + 0.5 * (probs - t) / (1 - t))


# ---------------------------------------------------------------------------
# Environment simulation
# ---------------------------------------------------------------------------
SCENARIOS = {
    # name: (has malicious traffic, has malware)
    "benign": (False, False),
    "network_attack": (True, False),
    "malware_drop": (False, True),
    "multi_stage": (True, True),
}


def simulate_environment(nids_pool, pe_pool, cfg):
    """
    Builds host sessions as chronological event streams.
    Pools are dicts {label: array of indices into the scored pool}.
    Multi-stage intrusions follow the kill chain: malicious traffic
    (recon / exploitation) precedes the malware landing on the host.
    """
    rng = np.random.default_rng(cfg["seed"])
    names = list(cfg["scenario_mix"].keys())
    weights = np.array([cfg["scenario_mix"][n] for n in names], dtype=float)
    scenario_draws = rng.choice(names, size=cfg["num_sessions"], p=weights / weights.sum())

    events = []
    for sid, scenario in enumerate(scenario_draws):
        net_attack, malware = SCENARIOS[scenario]
        n_net = rng.integers(*cfg["net_events_per_session"], endpoint=True)
        n_file = rng.integers(*cfg["file_events_per_session"], endpoint=True)

        net_labels = np.zeros(n_net, dtype=int)
        file_labels = np.zeros(n_file, dtype=int)
        if net_attack:
            n_mal = rng.integers(1, n_net + 1)
            net_labels[rng.choice(n_net, n_mal, replace=False)] = 1
        if malware:
            file_labels[rng.integers(n_file)] = 1

        stream = [("nids", int(l)) for l in net_labels] + [("pe", int(l)) for l in file_labels]
        rng.shuffle(stream)
        if scenario == "multi_stage":
            # Enforce kill-chain ordering: first malicious traffic before the malware drop
            first_net = next(i for i, e in enumerate(stream) if e == ("nids", 1))
            first_pe = next(i for i, e in enumerate(stream) if e == ("pe", 1))
            if first_pe < first_net:
                stream[first_net], stream[first_pe] = stream[first_pe], stream[first_net]

        for t, (domain, label) in enumerate(stream):
            pool = nids_pool if domain == "nids" else pe_pool
            events.append({
                "session_id": sid,
                "scenario": scenario,
                "t": t,
                "domain": domain,
                "pool_idx": int(rng.choice(pool[label])),
                "true_label": label,
            })
    return pd.DataFrame(events)


# ---------------------------------------------------------------------------
# Alerting & metrics
# ---------------------------------------------------------------------------
def alert_level(score, levels):
    for bound, name in levels:
        if score >= bound:
            return name
    return levels[-1][1]


def unified_score(r_nids, r_pe, cfg):
    """
    Fuses the per-domain risks into a single score in [0, 1].
    A Noisy-OR over every single event would saturate on long benign
    sessions, so each domain contributes only its strongest signal:
      - both detectors above threshold (corroborated) -> weighted Noisy-OR (boost)
      - otherwise                                     -> strongest weighted signal
    The score is monotone in both risks and crosses the alert threshold
    as soon as either detector fires.
    """
    w = cfg["domain_weights"]
    rn, rp = w["nids"] * r_nids, w["pe"] * r_pe
    t = cfg["alert_threshold"]
    if rn >= t and rp >= t:
        return 1.0 - (1.0 - rn) * (1.0 - rp)
    return max(rn, rp)


def aggregate_sessions(events, cfg):
    """Per-session unified score, alert flag and severity level."""
    rows = []
    for sid, g in events.groupby("session_id"):
        r_nids = g.loc[g["domain"] == "nids", "risk"].max()
        r_pe = g.loc[g["domain"] == "pe", "risk"].max()
        unified = unified_score(r_nids, r_pe, cfg)
        rows.append({
            "session_id": sid,
            "scenario": g["scenario"].iloc[0],
            "max_risk_nids": r_nids,
            "max_risk_pe": r_pe,
            "unified_score": unified,
            "alert": int(unified >= cfg["alert_threshold"]),
            "alert_level": alert_level(unified, cfg["alert_levels"]),
            "true_threat": int(g["true_label"].max()),
        })
    return pd.DataFrame(rows)


def calculate_impact_mitigation(events, sessions, cfg):
    """
    Defense-in-depth Impact Mitigation.
    For each threat session the timeline is walked chronologically and the
    first malicious event correctly flagged decides the outcome:
      - flagged at network level (traffic)  -> weight 'network' (attack contained before delivery)
      - flagged at host level (software)    -> weight 'host'    (malware reached the host)
      - never flagged                       -> weight 'missed'
    Pure network attacks have no host stage, so they are either stopped at
    the network or missed. False alerts on benign sessions are charged
    `false_alarm_cost` each, giving the net mitigation score.
    """
    mw = cfg["mitigation_weights"]
    outcomes = {}
    for sid, g in events[events["true_label"] == 1].sort_values("t").groupby("session_id"):
        hits = g[g["pred"] == 1]
        if hits.empty:
            outcomes[sid] = ("missed", mw["missed"])
        elif hits.iloc[0]["domain"] == "nids":
            outcomes[sid] = ("network", mw["network"])
        else:
            outcomes[sid] = ("host", mw["host"])

    sessions["stopped_at"] = sessions["session_id"].map(lambda s: outcomes.get(s, ("n/a", None))[0])
    threat_scores = [v for _, v in outcomes.values()]
    im = float(np.mean(threat_scores)) if threat_scores else 1.0

    benign = sessions[sessions["true_threat"] == 0]
    false_alarm_rate = float(benign["alert"].mean()) if len(benign) else 0.0
    n_threats = max(len(threat_scores), 1)
    im_net = (sum(threat_scores) - cfg["false_alarm_cost"] * benign["alert"].sum()) / n_threats
    return im, float(im_net), false_alarm_rate


def binary_metrics(y_true, y_pred, y_score):
    m = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan"),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    m.update({"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
              "fpr": fp / (fp + tn) if (fp + tn) else 0.0})
    return m


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def evaluate_unified_pipeline():
    cfg = get_unified_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Unified Pipeline] Initializing on {device}...")

    # 1. Data pools (test = unseen future, calibration = held-out past)
    nids_dir = project_root / CONFIG["cic_graphs_path"]
    _, nids_val_loader, nids_test_loader = get_nids_splits(nids_dir, batch_size=64, seed=cfg["seed"])
    nids_val, nids_test = nids_val_loader.dataset, nids_test_loader.dataset

    ember_dir = project_root / CONFIG["ember_graphs_path"]
    pe_test = sample_balanced(PEGraphDataset(str(ember_dir / "test")), cfg["pe_pool_per_class"], cfg["seed"])
    pe_calib = sample_balanced(PEGraphDataset(str(ember_dir / "train")), cfg["pe_calib_per_class"], cfg["seed"] + 1)

    nids_input_dim = nids_test[0].x.shape[1]
    pe_input_dim = pe_test[0].x.shape[1]
    print(f"[GNN] Input dims -> NIDS: {nids_input_dim} | PE: {pe_input_dim}")

    pe_model, nids_model = load_unified_system(device, pe_input_dim, nids_input_dim, cfg["pe_model"])

    # 2. Score every graph once, then calibrate each detector
    nids_cal_p, nids_cal_y = score_graphs(nids_model, nids_val, device, "Calibrating NIDS")
    pe_cal_p, pe_cal_y = score_graphs(pe_model, pe_calib, device, "Calibrating PE")
    nids_cal = ThresholdCalibrator("NIDS").fit(nids_cal_p, nids_cal_y)
    pe_cal = ThresholdCalibrator("PE").fit(pe_cal_p, pe_cal_y)

    nids_p, nids_y = score_graphs(nids_model, nids_test, device, "Scoring NIDS test pool")
    pe_p, pe_y = score_graphs(pe_model, pe_test, device, "Scoring PE test pool")
    scored = {
        "nids": {"prob": nids_p, "risk": nids_cal.transform(nids_p)},
        "pe": {"prob": pe_p, "risk": pe_cal.transform(pe_p)},
    }
    nids_pool = {l: np.where(nids_y == l)[0] for l in (0, 1)}
    pe_pool = {l: np.where(pe_y == l)[0] for l in (0, 1)}
    for name, pool in (("NIDS", nids_pool), ("PE", pe_pool)):
        print(f"[Pools] {name}: benign={len(pool[0])} malicious={len(pool[1])}")
        if min(len(p) for p in pool.values()) == 0:
            raise RuntimeError(f"{name} test pool is missing one class; cannot simulate mixed scenarios.")

    # 3. Simulate the monitored environment and classify every event
    events = simulate_environment(nids_pool, pe_pool, cfg)
    events["prob"] = [scored[d]["prob"][i] for d, i in zip(events["domain"], events["pool_idx"])]
    events["risk"] = [scored[d]["risk"][i] for d, i in zip(events["domain"], events["pool_idx"])]
    events["pred"] = (events["risk"] >= 0.5).astype(int)
    events["verdict"] = np.where(events["pred"] == 1, "MALICIOUS", "BENIGN")

    # 4. Aggregate into unified per-session alerts
    sessions = aggregate_sessions(events, cfg)
    im, im_net, far = calculate_impact_mitigation(events, sessions, cfg)

    # 5. Metrics
    event_metrics = {d: binary_metrics(g["true_label"], g["pred"], g["risk"])
                     for d, g in events.groupby("domain")}
    session_metrics = binary_metrics(sessions["true_threat"], sessions["alert"], sessions["unified_score"])

    print("\n" + "=" * 60)
    print(" UNIFIED ALERTING SYSTEM RESULTS")
    print("=" * 60)
    print(f"Sessions: {len(sessions)} | Events: {len(events)} "
          f"(network windows: {(events['domain'] == 'nids').sum()}, software: {(events['domain'] == 'pe').sum()})")
    print("\nEvent-level classification:")
    for d, label in (("nids", "Network packets/windows"), ("pe", "Software (PE files)")):
        m = event_metrics[d]
        print(f" - {label:<24} AUC {m['auc']:.4f} | F1 {m['f1']:.4f} | "
              f"Recall {m['recall']:.2%} | FPR {m['fpr']:.2%}")
    m = session_metrics
    print("\nSession-level unified score:")
    print(f" - AUC {m['auc']:.4f} | F1 {m['f1']:.4f} | Precision {m['precision']:.2%} | "
          f"Recall {m['recall']:.2%} | FPR {m['fpr']:.2%}")
    print(f" - Confusion: TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}")
    print("\nImpact Mitigation:")
    print(f" - Gross IM:              {im:.2%}")
    print(f" - Net IM (false alarms): {im_net:.2%}  (false alarm rate {far:.2%})")

    print("\nBreakdown by scenario:")
    for scenario, g in sessions.groupby("scenario"):
        rate = g["alert"].mean()
        if g["true_threat"].iloc[0] == 1:
            stops = g["stopped_at"].value_counts(normalize=True)
            stop_str = ", ".join(f"{k}: {v:.0%}" for k, v in stops.items())
            print(f" - {scenario:<15} detection {rate:.2%} | stopped at -> {stop_str}")
        else:
            print(f" - {scenario:<15} false positive rate {rate:.2%}")

    print("\nAlert level distribution:")
    print(pd.crosstab(sessions["scenario"], sessions["alert_level"]).to_string())

    # 6. Persist artifacts
    out_dir = project_root / CONFIG["experiments_path"] / "unified_pipeline" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(out_dir / "events.csv", index=False)
    sessions.to_csv(out_dir / "sessions.csv", index=False)
    summary = {
        "config": {k: v for k, v in cfg.items() if k != "alert_levels"},
        "thresholds": {"nids": nids_cal.threshold, "pe": pe_cal.threshold},
        "event_metrics": event_metrics,
        "session_metrics": session_metrics,
        "impact_mitigation": im,
        "impact_mitigation_net": im_net,
        "false_alarm_rate": far,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n[Success] Unified pipeline artifacts saved to: {out_dir}")


if __name__ == "__main__":
    evaluate_unified_pipeline()
