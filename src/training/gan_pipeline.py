"""
Shared GAN augmentation pipeline for both domains (PE / EMBER and NIDS / CIC-IDS2017):

    features -> graph -> [1] Dropout-WGAN-GP on malicious graphs only
                      -> [2] synthetic malicious graphs (label 1)
                      -> [3] MFGraph trained on real + synthetic
                      -> [4] final score on the held-out test set(s)
"""
import os
import copy
import json
import random
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Subset, ConcatDataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

from src.models.gan import GraphGenerator, GraphDiscriminator
from src.models.gnn import MFGraph
from src.utils.config import CONFIG

gnn_config = CONFIG["gnn"]
gan_config = CONFIG["gan"]


def num_workers():
    return max(0, min(8, (os.cpu_count() or 6) - 1))


def make_loader(dataset, batch_size, shuffle):
    workers = num_workers()
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, persistent_workers=workers > 0)


def scan_train_set(train_data, per_node_range: bool):
    """
    Single pass over the training split: returns graph labels, the index of
    malicious graphs and the real feature range of the malicious class.
    """
    loader = make_loader(train_data, batch_size=512, shuffle=False)
    labels, feat_min, feat_max = [], None, None
    num_nodes = None

    for batch in tqdm(loader, desc="[Data] Scanning training set", unit="batch"):
        y = batch.y.view(-1)
        labels.append(y)
        mal = y == 1
        if not mal.any():
            continue

        x = batch.x[mal[batch.batch]]
        graph_size = int(torch.bincount(batch.batch)[0])
        num_nodes = graph_size if num_nodes is None else num_nodes
        if per_node_range:
            x = x.view(-1, graph_size, x.size(-1))
            b_min, b_max = x.min(0).values, x.max(0).values
        else:
            b_min, b_max = x.min(0, keepdim=True).values, x.max(0, keepdim=True).values

        feat_min = b_min if feat_min is None else torch.minimum(feat_min, b_min)
        feat_max = b_max if feat_max is None else torch.maximum(feat_max, b_max)

    labels = torch.cat(labels)
    mal_idx = torch.nonzero(labels == 1).view(-1).tolist()
    print(f"[Data] Train graphs: {len(labels)} (Benign: {(labels == 0).sum().item()}, "
          f"Malicious: {len(mal_idx)}) | nodes/graph: {num_nodes}")
    return labels, mal_idx, feat_min, feat_max, num_nodes


def gradient_penalty(critic, real_x, fake_x, edge_index, batch):
    """WGAN-GP penalty, computed per graph on interpolated node features."""
    num_graphs = int(batch.max().item()) + 1
    eps = torch.rand(num_graphs, 1, device=real_x.device)[batch]
    x_hat = (eps * real_x + (1 - eps) * fake_x).requires_grad_(True)

    score = critic(x_hat, edge_index, batch)
    grad = torch.autograd.grad(score.sum(), x_hat, create_graph=True)[0]

    sq_norm = torch.zeros(num_graphs, device=real_x.device).index_add_(0, batch, grad.pow(2).sum(-1))
    return ((sq_norm + 1e-12).sqrt() - 1).pow(2).mean()


def train_wgan(mal_loader, generator, input_dim, device, logger):
    """
    Dropout-GAN with WGAN-GP critics: at each step every critic is dropped with
    probability `dropout_threshold`; the generator is trained against the active ones.
    """
    critics = [GraphDiscriminator(input_dim=input_dim, hidden_dim=gnn_config["hidden_dim"],
                                  k=gnn_config["k"], dropout_rate=0.0).to(device)
               for _ in range(gan_config["num_discriminators"])]

    betas = (0.5, 0.9)
    opt_g = torch.optim.Adam(generator.parameters(), lr=gan_config["lr_generator"], betas=betas)
    opts_d = [torch.optim.Adam(c.parameters(), lr=gan_config["lr_discriminator"], betas=betas) for c in critics]

    n_critic = gan_config["n_critic"]
    gp_lambda = gan_config["gp_lambda"]
    dropout_threshold = gan_config["dropout_threshold"]
    gan_epochs = gan_config["epochs"]

    generator.train()
    for c in critics: c.train()

    step = 0
    for epoch in range(1, gan_epochs + 1):
        sums = {"d_loss": 0.0, "g_loss": 0.0, "w_dist": 0.0, "gp": 0.0, "feat_mean_gap": 0.0}
        d_steps, g_steps = 0, 0

        pbar = tqdm(mal_loader, desc=f"[WGAN-GP] Epoch {epoch}/{gan_epochs}", unit="batch")
        for batch in pbar:
            batch = batch.to(device)
            ei, b = batch.edge_index, batch.batch

            active = [i for i in range(len(critics)) if random.random() > dropout_threshold]
            if not active:
                active = [random.randrange(len(critics))]

            # --- Critic step ---
            with torch.no_grad():
                fake_x = generator(ei, b)

            d_loss_step, w_step, gp_step = 0.0, 0.0, 0.0
            for i in active:
                opts_d[i].zero_grad()
                real_score = critics[i](batch.x, ei, b).mean()
                fake_score = critics[i](fake_x, ei, b).mean()
                gp = gradient_penalty(critics[i], batch.x, fake_x, ei, b)
                loss_d = fake_score - real_score + gp_lambda * gp
                loss_d.backward()
                opts_d[i].step()

                d_loss_step += loss_d.item()
                w_step += (real_score - fake_score).item()
                gp_step += gp.item()

            sums["d_loss"] += d_loss_step / len(active)
            sums["w_dist"] += w_step / len(active)
            sums["gp"] += gp_step / len(active)
            sums["feat_mean_gap"] += (batch.x.mean(0) - fake_x.mean(0)).abs().mean().item()
            d_steps += 1
            step += 1

            # --- Generator step (every n_critic critic steps) ---
            if step % n_critic == 0:
                opt_g.zero_grad()
                fake_x = generator(ei, b)
                loss_g = -sum(critics[i](fake_x, ei, b).mean() for i in active) / len(active)
                loss_g.backward()
                opt_g.step()
                sums["g_loss"] += loss_g.item()
                g_steps += 1

            pbar.set_postfix(W=f"{w_step / len(active):.3f}", GP=f"{gp_step / len(active):.3f}")

        metrics = {
            "d_loss": sums["d_loss"] / max(d_steps, 1),
            "g_loss": sums["g_loss"] / max(g_steps, 1),
            "w_dist": sums["w_dist"] / max(d_steps, 1),
            "gp": sums["gp"] / max(d_steps, 1),
            "feat_mean_gap": sums["feat_mean_gap"] / max(d_steps, 1),
        }
        logger.log_gan_epoch(epoch, metrics)
        print(f"[WGAN-GP] Epoch [{epoch}/{gan_epochs}] | " +
              " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    torch.save(generator.state_dict(), logger.run_dir / "generator.pth")
    return generator


def generate_synthetic(generator, mal_data, num_synthetic, device):
    """Generates malicious graphs on the topology of real malicious training graphs."""
    generator.eval()
    synthetic = []
    loader = DataLoader(mal_data, batch_size=256, shuffle=True)

    with torch.no_grad(), tqdm(total=num_synthetic, desc="[Synth] Generating malicious graphs") as pbar:
        while len(synthetic) < num_synthetic:
            for batch in loader:
                batch = batch.to(device)
                batch.x = generator(batch.edge_index, batch.batch)
                graphs = batch.cpu().to_data_list()[: num_synthetic - len(synthetic)]
                for g in graphs:
                    g.y = torch.ones_like(g.y)
                synthetic.extend(graphs)
                pbar.update(len(graphs))
                if len(synthetic) >= num_synthetic:
                    break
    return synthetic


def evaluate_loader(loader, model, criterion, device, desc):
    all_preds, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, unit="batch", leave=False):
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch).squeeze(-1)
            y = batch.y.view(-1).float()
            total_loss += criterion(logits, y).item() * batch.num_graphs
            all_preds.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_labels = np.array(all_labels)
    bin_preds = (np.array(all_preds) >= 0.5).astype(int)
    auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else float("nan")

    return {
        "loss": total_loss / len(loader.dataset),
        "auc": auc,
        "accuracy": accuracy_score(all_labels, bin_preds),
        "f1": f1_score(all_labels, bin_preds, zero_division=0),
        "precision": precision_score(all_labels, bin_preds, zero_division=0),
        "recall": recall_score(all_labels, bin_preds, zero_division=0),
    }


def train_classifier(train_dataset, num_benign, num_malicious, val_loader, input_dim, lr, device, logger):
    """MFGraph trained on real + synthetic graphs; best epoch selected on validation AUC."""
    train_loader = make_loader(train_dataset, gnn_config["batch_size"], shuffle=True)

    model = MFGraph(input_dim=input_dim, hidden_dim=gnn_config["hidden_dim"],
                    k=gnn_config["k"], dropout_rate=gnn_config["dropout_rate"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

    pos_weight = torch.tensor([num_benign / max(num_malicious, 1)], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"[MFGraph] Augmented set — Benign: {num_benign}, Malicious: {num_malicious} "
          f"| pos_weight: {pos_weight.item():.4f}")

    best_auc, best_weights = -1.0, copy.deepcopy(model.state_dict())
    epochs = gnn_config["num_epochs"]

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[MFGraph] Epoch {epoch:02d}/{epochs} [Train]", unit="batch")
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.batch).squeeze(-1)
            loss = criterion(logits, batch.y.view(-1).float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs
            pbar.set_postfix(Loss=f"{loss.item():.4f}")

        model.eval()
        val = evaluate_loader(val_loader, model, criterion, device, f"[MFGraph] Epoch {epoch:02d} [Val]")
        scheduler.step(val["auc"])

        metrics = {"train_loss": train_loss / len(train_dataset), "test_loss": val.pop("loss"), **val}
        logger.log_epoch(epoch=epoch, metrics=metrics)
        print(f"Epoch {epoch:02d}/{epochs} - Train Loss: {metrics['train_loss']:.4f} | "
              f"Val Loss: {metrics['test_loss']:.4f} | AUC: {metrics['auc']:.4f} | F1: {metrics['f1']:.4f}")

        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            best_weights = copy.deepcopy(model.state_dict())
            print(f"  -> New best validation AUC: {best_auc:.4f}")

    model.load_state_dict(best_weights)
    torch.save(best_weights, logger.run_dir / "model_gan.pth")
    return model, criterion


def run_gan_pipeline(train_data, val_loader, test_loaders: dict, input_dim, per_node_range, lr, logger):
    """
    Args:
        train_data: training split (real graphs, benign + malicious).
        val_loader: validation loader used for model selection.
        test_loaders: {name: loader} evaluated with the final model (e.g. one per month).
        per_node_range: True when node i always has the same meaning (PE feature groups).
        lr: MFGraph learning rate.
    """
    torch.manual_seed(CONFIG["random_seed"])
    random.seed(CONFIG["random_seed"])
    torch.set_num_threads(os.cpu_count() or 6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels, mal_idx, feat_min, feat_max, num_nodes = scan_train_set(train_data, per_node_range)
    mal_data = Subset(train_data, mal_idx)

    print("\n--- [Phase 1/4] Dropout-WGAN-GP Training (malicious graphs only) ---")
    generator = GraphGenerator(noise_dim=gan_config["noise_dim"], output_dim=input_dim, num_nodes=num_nodes,
                               feat_min=feat_min, feat_max=feat_max).to(device)
    mal_loader = make_loader(mal_data, gan_config["batch_size"], shuffle=True)
    generator = train_wgan(mal_loader, generator, input_dim, device, logger)

    print("\n--- [Phase 2/4] Synthetic Malicious Graph Generation ---")
    num_synthetic = int(gan_config["synthetic_ratio"] * len(mal_idx))
    synthetic = generate_synthetic(generator, mal_data, num_synthetic, device)
    augmented = ConcatDataset([train_data, synthetic])

    print("\n--- [Phase 3/4] MFGraph Classifier Training (real + synthetic) ---")
    num_benign = int((labels == 0).sum())
    model, criterion = train_classifier(augmented, num_benign, len(mal_idx) + len(synthetic),
                                        val_loader, input_dim, lr, device, logger)

    print("\n--- [Phase 4/4] Final Score ---")
    final_scores = {}
    for name, loader in test_loaders.items():
        scores = evaluate_loader(loader, model, criterion, device, f"[Test] {name}")
        scores.pop("loss")
        final_scores[name] = scores
        logger.log_concept_drift(name, scores)
        print(f"[Test] {name}: AUC={scores['auc']:.4f} | F1={scores['f1']:.4f} | "
              f"Acc={scores['accuracy']:.4f} | Prec={scores['precision']:.4f} | Rec={scores['recall']:.4f}")

    if len(final_scores) > 1:
        final_scores["mean"] = {k: float(np.nanmean([s[k] for s in final_scores.values()]))
                                for k in next(iter(final_scores.values()))}
        print(f"[Test] mean: AUC={final_scores['mean']['auc']:.4f} | F1={final_scores['mean']['f1']:.4f}")

    with open(logger.run_dir / "final_scores.json", "w") as f:
        json.dump(final_scores, f, indent=2)

    print(f"\n[Success] GAN pipeline completed! Results in: {logger.run_dir}")
    return final_scores
