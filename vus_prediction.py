# -*- coding: utf-8 -*-
"""
Applies the already-trained, tuned PAE-weighted GATv2 model (Phase 4) to real ClinVar
Variants of Uncertain Significance (VUS), the variant class that motivated this project's
introduction but was explicitly excluded from training and evaluation everywhere else in the
paper (Phase 1 keeps only variants with an explicit Pathogenic/Benign label).

This is an exploratory, descriptive analysis, not a validation: VUS have no ground-truth label,
so there is nothing to score these predictions against. What this script produces is a frozen
record of what the trained model outputs when pointed at the real-world backlog it was motivated
by, restricted to the same 6,597 proteins already used throughout the project (so no new
AlphaFold/ESM-2/UniProt downloads are needed, every VUS considered here sits on a protein
that already has a cached structural graph).

Outputs (both permanent, re-running this script later will NOT overwrite them or silently
change the numbers, since ClinVar's public dump is revised over time and re-classifications
would otherwise make results drift):
    vus_predictions.csv        - one row per scored VUS: identifiers + predicted probability
    vus_analysis_summary.json  - descriptive statistics + full provenance (data snapshot used,
                                  model checkpoint used, exact run timestamp)

To force a genuine re-run (e.g. against a freshly downloaded variant_summary.txt.gz), delete
both output files first.
"""
import os
import re
import json
import hashlib
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler
from torch_geometric.loader import DataLoader

from missense_dataset import MissenseVariantGraphDataset, NODE_FEATURE_DIM
from models import GATv2Classifier

PRED_OUT = "vus_predictions.csv"
SUMMARY_OUT = "vus_analysis_summary.json"
CLINVAR_RAW = "variant_summary.txt.gz"
GRAPH_DIR = "protein_graphs"
META_DIR = os.path.join("alphafold_data", "metadata")
EXISTING_DATASET = "clinvar_graph_ready.csv"
CHECKPOINT = os.path.join("models", "gatv2_pae_weighted_final_best.pt")
BEST_CONFIG = "best_config.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AA_3TO1 = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C', 'Gln': 'Q', 'Glu': 'E', 'Gly': 'G',
    'His': 'H', 'Ile': 'I', 'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P', 'Ser': 'S',
    'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V',
}

# Same explicit, unambiguous VUS labels as ClinVar's own convention - excludes anything mixed
# with a pathogenic/benign call or a different qualifier, for the same reason Phase 1 excludes
# "Conflicting interpretations": training/scoring on genuinely ambiguous labels is not useful.
VUS_LABELS = ["Uncertain significance", "Uncertain significance/Uncertain risk allele"]

VALID_REVIEW_STATUSES = [
    "criteria provided, multiple submitters, no conflicts",
    "reviewed by expert panel",
    "practice guideline",
]


def file_sha256_head(path, n_bytes=8_000_000):
    """Hash of the first n_bytes only - enough to fingerprint the file without hashing 420MB+."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(n_bytes))
    return h.hexdigest()


def parse_missense(name_str):
    """Identical HGVS parsing logic to Phase 1 - kept in lockstep on purpose."""
    if pd.isna(name_str):
        return pd.Series({"Ref_AA": None, "Pos": None, "Alt_AA": None})
    match = re.search(r"\(p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|\*|Ter|=)\)", name_str)
    if match:
        ref_aa, pos, alt_aa = match.groups()
        if ref_aa != alt_aa and alt_aa not in ["Ter", "*", "=", "Del", "Dup"]:
            return pd.Series({"Ref_AA": ref_aa, "Pos": int(pos), "Alt_AA": alt_aa})
    return pd.Series({"Ref_AA": None, "Pos": None, "Alt_AA": None})


class NodeBudgetBatchSampler(Sampler):
    """Identical to the sampler used in Phase 4/5 - caps total residue count per batch."""

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


def main():
    if os.path.exists(PRED_OUT) and os.path.exists(SUMMARY_OUT):
        print(f"{PRED_OUT} and {SUMMARY_OUT} already exist - loading the frozen results instead "
              f"of recomputing. Delete both files first to force a genuine re-run.")
        summary = json.load(open(SUMMARY_OUT))
        print(json.dumps(summary, indent=2))
        return

    run_started = datetime.now(timezone.utc).isoformat()
    print(f"Run started (UTC): {run_started}")
    print(f"Device: {DEVICE}")

    if not os.path.exists(CLINVAR_RAW):
        raise FileNotFoundError(
            f"{CLINVAR_RAW} not found. This script deliberately does not auto-download it, "
            f"since a freshly downloaded copy would reflect ClinVar's *current* classifications, "
            f"not the ones the rest of this project was built on."
        )
    clinvar_fingerprint = file_sha256_head(CLINVAR_RAW)
    clinvar_mtime = datetime.fromtimestamp(os.path.getmtime(CLINVAR_RAW), tz=timezone.utc).isoformat()
    print(f"Using {CLINVAR_RAW} (mtime {clinvar_mtime}, head-SHA256 {clinvar_fingerprint[:16]}...)")

    # ---- restrict to genes already covered by this project's 6,597 proteins ----
    existing = pd.read_csv(EXISTING_DATASET, usecols=["GeneSymbol", "UniProt_ID", "Pos", "Alt_AA_1"])
    gene_to_uid = existing.drop_duplicates("GeneSymbol").set_index("GeneSymbol")["UniProt_ID"].to_dict()
    existing_variant_keys = set(zip(existing["UniProt_ID"], existing["Pos"], existing["Alt_AA_1"]))
    print(f"Genes already covered by this project: {len(gene_to_uid):,}")

    # ---- load and filter the raw ClinVar dump ----
    cols = ["VariationID", "Type", "Name", "GeneSymbol", "ClinicalSignificance",
            "ReviewStatus", "OriginSimple", "Assembly"]
    df = pd.read_csv(CLINVAR_RAW, sep="\t", compression="gzip", dtype=str, usecols=cols, low_memory=False)
    print(f"Raw ClinVar rows: {len(df):,}")

    df = df[df["Assembly"] == "GRCh38"]
    df = df[df["Type"] == "single nucleotide variant"]
    df = df[df["OriginSimple"] == "germline"]
    df = df[df["ReviewStatus"].isin(VALID_REVIEW_STATUSES)]
    df = df[df["ClinicalSignificance"].isin(VUS_LABELS)]
    print(f"VUS rows after GRCh38/SNV/germline/2-star-review/label filters: {len(df):,}")

    df = df[df["GeneSymbol"].isin(gene_to_uid)].copy()
    df["UniProt_ID"] = df["GeneSymbol"].map(gene_to_uid)
    print(f"VUS rows restricted to genes already in this project's protein set: {len(df):,}")

    df = df.drop_duplicates(subset=["VariationID"]).copy()
    print(f"After VariationID dedup: {len(df):,}")

    parsed = df["Name"].apply(parse_missense)
    df = pd.concat([df, parsed], axis=1)
    df = df.dropna(subset=["Ref_AA"]).copy()
    df["Pos"] = df["Pos"].astype(int)
    print(f"After missense (HGVS) extraction: {len(df):,}")

    df["Alt_AA_1"] = df["Alt_AA"].map(AA_3TO1)
    df["Ref_AA_1"] = df["Ref_AA"].map(AA_3TO1)
    df = df.dropna(subset=["Alt_AA_1", "Ref_AA_1"]).copy()

    # ---- validate against the same cached canonical sequence used to build the graphs ----
    seq_cache = {}

    def validate(row):
        uid = row["UniProt_ID"]
        if uid not in seq_cache:
            meta_path = os.path.join(META_DIR, f"{uid}.json")
            graph_path = os.path.join(GRAPH_DIR, f"{uid}.pt")
            if not (os.path.exists(meta_path) and os.path.exists(graph_path)):
                seq_cache[uid] = None
            else:
                with open(meta_path) as f:
                    seq_cache[uid] = json.load(f)["uniprotSequence"]
        seq = seq_cache[uid]
        if seq is None:
            return False
        pos = int(row["Pos"])
        if pos < 1 or pos > len(seq):
            return False
        return seq[pos - 1] == row["Ref_AA_1"]

    df["Is_Valid"] = df.apply(validate, axis=1)
    df = df[df["Is_Valid"]].copy()
    print(f"After reference-residue validation against cached sequences: {len(df):,}")

    # ---- purity check: make sure no VUS row collides with an existing labelled variant ----
    df["_key"] = list(zip(df["UniProt_ID"], df["Pos"], df["Alt_AA_1"]))
    overlap = df["_key"].isin(existing_variant_keys)
    if overlap.any():
        print(f"Dropping {overlap.sum():,} VUS rows that collide with an existing labelled "
              f"(Pathogenic/Benign) variant already in {EXISTING_DATASET}.")
        df = df[~overlap].copy()
    df = df.drop(columns=["_key"])
    df = df.drop_duplicates(subset=["UniProt_ID", "Pos", "Alt_AA_1"]).reset_index(drop=True)
    print(f"Final VUS set to score: {len(df):,} variants across {df['UniProt_ID'].nunique():,} proteins")

    if len(df) == 0:
        raise RuntimeError("No VUS variants survived filtering - nothing to score.")

    # ---- load the tuned PAE-weighted checkpoint, exactly as Phase 5 does ----
    with open(BEST_CONFIG) as f:
        best_config = json.load(f)
    print(f"best_config: {best_config['name']}  hidden_dim={int(best_config['hidden_dim'])}  "
          f"dropout={best_config['dropout']}")

    model = GATv2Classifier(in_dim=NODE_FEATURE_DIM, use_edge_attr=True,
                             hidden_dim=int(best_config["hidden_dim"]),
                             dropout=best_config["dropout"])
    model.load_state_dict(torch.load(CHECKPOINT, weights_only=True))
    model.to(DEVICE)
    model.eval()
    print(f"Loaded {CHECKPOINT}")

    df["Target"] = -1  # placeholder: VUS has no ground truth, never used for scoring

    ds = MissenseVariantGraphDataset(df, GRAPH_DIR)
    graph_cache = {}
    node_counts = []
    for uid in df["UniProt_ID"]:
        if uid not in graph_cache:
            g = torch.load(os.path.join(GRAPH_DIR, f"{uid}.pt"), weights_only=False)
            graph_cache[uid] = g.num_nodes if hasattr(g, "num_nodes") else g.x.shape[0]
        node_counts.append(graph_cache[uid])

    loader = DataLoader(ds, batch_sampler=NodeBudgetBatchSampler(node_counts))

    all_probs = []
    t0 = time.time()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = batch.to(DEVICE)
            logits = model(batch)
            all_probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            if (i + 1) % 20 == 0:
                print(f"  batch {i + 1}/{len(loader)}  ({len(all_probs):,} variants scored)")
    print(f"Inference done in {time.time() - t0:.1f}s")

    df["predicted_probability"] = all_probs
    df["predicted_label"] = np.where(df["predicted_probability"] >= 0.5,
                                      "likely pathogenic (model)", "likely benign (model)")

    out_cols = ["VariationID", "GeneSymbol", "UniProt_ID", "Ref_AA", "Pos", "Alt_AA",
                "predicted_probability", "predicted_label"]
    df[out_cols].to_csv(PRED_OUT, index=False)
    print(f"Saved {len(df):,} predictions to {PRED_OUT}")

    probs = df["predicted_probability"].values
    summary = {
        "run_started_utc": run_started,
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "device_used": DEVICE,
        "clinvar_dump_file": CLINVAR_RAW,
        "clinvar_dump_mtime_utc": clinvar_mtime,
        "clinvar_dump_head_sha256": clinvar_fingerprint,
        "model_checkpoint": CHECKPOINT,
        "best_config": best_config,
        "n_vus_scored": int(len(df)),
        "n_proteins_covered": int(df["UniProt_ID"].nunique()),
        "n_genes_covered": int(df["GeneSymbol"].nunique()),
        "pct_predicted_pathogenic_at_0.5": float((probs >= 0.5).mean() * 100),
        "pct_predicted_benign_at_0.5": float((probs < 0.5).mean() * 100),
        "mean_predicted_probability": float(probs.mean()),
        "median_predicted_probability": float(np.median(probs)),
        "std_predicted_probability": float(probs.std()),
        "note": (
            "Descriptive only. VUS have no ground-truth label, so these predictions cannot be "
            "validated the way every other result in this project was. This summary and "
            "vus_predictions.csv are the frozen, authoritative record of this run; re-running "
            "this script after ClinVar publishes a new variant_summary.txt.gz would see a "
            "different, larger, and partly reclassified VUS pool, which is exactly why these "
            "outputs are saved rather than recomputed on demand."
        ),
    }
    with open(SUMMARY_OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {SUMMARY_OUT}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
