# -*- coding: utf-8 -*-
"""One-off script: computes real training-set AUPRC/AUROC/F1 for all seven models in
Table I, to add a train-set column to the table (supervisor request). Reuses the exact
protein-level split function and model architectures from phase4_model_training.ipynb.
Also re-evaluates each model on the test set as a sanity check, to confirm the loaded
checkpoints and reconstructed split reproduce the numbers already published in the paper
before trusting the new train-set numbers."""
import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler
from torch_geometric.loader import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

from missense_dataset import MissenseVariantGraphDataset, NODE_FEATURE_DIM, ESM2_EMBEDDING_DIM, AA_1TO_ONEHOT_IDX, N_AA
from models import GATv2Classifier, GCNClassifier, GraphSAGEClassifier

GRAPH_DIR = "protein_graphs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_protein_split(df, seed, train_frac=0.70, val_frac=0.15):
    rng = np.random.RandomState(seed)
    unique_proteins = df["UniProt_ID"].unique().copy()
    rng.shuffle(unique_proteins)
    n = len(unique_proteins)
    n_train, n_val = int(n * train_frac), int(n * val_frac)
    train_p = set(unique_proteins[:n_train])
    val_p = set(unique_proteins[n_train:n_train + n_val])
    test_p = set(unique_proteins[n_train + n_val:])
    train_df = df[df["UniProt_ID"].isin(train_p)].reset_index(drop=True)
    val_df = df[df["UniProt_ID"].isin(val_p)].reset_index(drop=True)
    test_df = df[df["UniProt_ID"].isin(test_p)].reset_index(drop=True)
    return train_df, val_df, test_df


class NodeBudgetBatchSampler(Sampler):
    def __init__(self, node_counts, max_nodes_per_batch=20000):
        self.node_counts = node_counts
        self.max_nodes_per_batch = max_nodes_per_batch

    def __iter__(self):
        batch, current = [], 0
        for idx in range(len(self.node_counts)):
            n = self.node_counts[idx]
            if batch and current + n > self.max_nodes_per_batch:
                yield batch
                batch, current = [], 0
            batch.append(idx)
            current += n
        if batch:
            yield batch

    def __len__(self):
        return max(1, sum(self.node_counts) // self.max_nodes_per_batch)


def get_node_counts(variant_df, graph_dir):
    cache, counts = {}, []
    for uid in variant_df["UniProt_ID"]:
        if uid not in cache:
            g = torch.load(os.path.join(graph_dir, f"{uid}.pt"), weights_only=False)
            cache[uid] = g.num_nodes
        counts.append(cache[uid])
    return counts


def evaluate_gnn(model, variant_df, desc):
    ds = MissenseVariantGraphDataset(variant_df, GRAPH_DIR)
    nc = get_node_counts(variant_df, GRAPH_DIR)
    loader = DataLoader(ds, batch_sampler=NodeBudgetBatchSampler(nc))
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            logits.extend(model(batch).cpu().numpy().tolist())
            labels.extend(batch.y.cpu().numpy().tolist())
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    labels = np.array(labels)
    preds = (probs >= 0.5).astype(int)
    auprc = average_precision_score(labels, probs)
    auroc = roc_auc_score(labels, probs)
    f1 = f1_score(labels, preds)
    print(f"  {desc}: AUPRC={auprc:.4f} AUROC={auroc:.4f} F1={f1:.4f}  (n={len(labels)})")
    return auprc, auroc, f1


def extract_flat_features(variant_df, graph_dir):
    cache = {}
    rows = []
    for _, row in variant_df.iterrows():
        uid = row["UniProt_ID"]
        if uid not in cache:
            cache[uid] = torch.load(os.path.join(graph_dir, f"{uid}.pt"), weights_only=False)
            if len(cache) > 64:
                cache.pop(next(iter(cache)))
        g = cache[uid]
        node_idx = int(row["Pos"]) - 1
        esm_emb = g.x[node_idx, :ESM2_EMBEDDING_DIM].numpy()
        wt_onehot = g.x[node_idx, ESM2_EMBEDDING_DIM:ESM2_EMBEDDING_DIM + N_AA].numpy()
        alt_onehot = np.zeros(N_AA, dtype=np.float32)
        alt_onehot[AA_1TO_ONEHOT_IDX[row["Alt_AA_1"]]] = 1.0
        rows.append(np.concatenate([esm_emb, wt_onehot, alt_onehot]))
    return np.stack(rows)


def main():
    df = pd.read_csv("clinvar_graph_ready.csv")
    train_df, val_df, test_df = make_protein_split(df, seed=42)
    print(f"Split: train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")
    assert len(test_df) == 6771, f"Split reconstruction mismatch: expected 6771 test variants, got {len(test_df)}"
    print("Split reconstruction verified against paper's reported test-set size (6,771).\n")

    results = {}

    configs = [
        ("GraphSAGE (baseline)", GraphSAGEClassifier, dict(), "models/graphsage_baseline_best.pt"),
        ("XGBoost (no structure)", None, None, None),  # handled separately
        ("GATv2 (PAE-w., original)", GATv2Classifier, dict(use_edge_attr=True), "models/gatv2_pae_weighted_best.pt"),
        ("GATv2 (unweighted, original)", GATv2Classifier, dict(use_edge_attr=False), "models/gatv2_unweighted_best.pt"),
        ("GATv2 (unweighted, tuned)", GATv2Classifier, dict(use_edge_attr=False, hidden_dim=64, dropout=0.4), "models/gatv2_unweighted_final_best.pt"),
        ("GATv2 (PAE-w., tuned)", GATv2Classifier, dict(use_edge_attr=True, hidden_dim=64, dropout=0.4), "models/gatv2_pae_weighted_final_best.pt"),
        ("GCN (baseline)", GCNClassifier, dict(), "models/gcn_baseline_best.pt"),
    ]

    for name, cls, kwargs, ckpt in configs:
        if cls is None:
            continue
        print(f"=== {name} ===")
        model = cls(in_dim=NODE_FEATURE_DIM, **kwargs)
        model.load_state_dict(torch.load(ckpt, weights_only=True))
        model.to(DEVICE)
        train_metrics = evaluate_gnn(model, train_df, "train")
        test_metrics = evaluate_gnn(model, test_df, "test (sanity check)")
        results[name] = {"train": train_metrics, "test": test_metrics}
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        print()

    print("=== XGBoost (no structure) ===")
    import xgboost as xgb
    X_train = extract_flat_features(train_df, GRAPH_DIR)
    X_val = extract_flat_features(val_df, GRAPH_DIR)
    X_test = extract_flat_features(test_df, GRAPH_DIR)
    y_train = train_df["Target"].values
    y_val = val_df["Target"].values
    y_test = test_df["Target"].values

    train_pos = (train_df["Target"] == 1).sum()
    train_neg = (train_df["Target"] == 0).sum()
    pos_weight_value = train_neg / train_pos

    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        scale_pos_weight=pos_weight_value, eval_metric="logloss", early_stopping_rounds=20, random_state=42,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    def xgb_metrics(X, y, desc):
        probs = xgb_model.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)
        auprc = average_precision_score(y, probs)
        auroc = roc_auc_score(y, probs)
        f1 = f1_score(y, preds)
        print(f"  {desc}: AUPRC={auprc:.4f} AUROC={auroc:.4f} F1={f1:.4f}  (n={len(y)})")
        return auprc, auroc, f1

    train_metrics = xgb_metrics(X_train, y_train, "train")
    test_metrics = xgb_metrics(X_test, y_test, "test (sanity check)")
    results["XGBoost (no structure)"] = {"train": train_metrics, "test": test_metrics}

    with open("table1_train_test_metrics.json", "w") as f:
        json.dump({k: {"train": list(v["train"]), "test": list(v["test"])} for k, v in results.items()}, f, indent=2)
    print("\nSaved table1_train_test_metrics.json")
    print(json.dumps({k: {"train": list(v["train"]), "test": list(v["test"])} for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
