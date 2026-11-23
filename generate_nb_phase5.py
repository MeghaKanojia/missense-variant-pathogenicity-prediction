import json, uuid, os

def cid():
    return str(uuid.uuid4())[:8]

def code(*lines):
    return {"cell_type": "code", "execution_count": None, "id": cid(),
            "metadata": {}, "outputs": [], "source": list(lines)}

def md(*lines):
    return {"cell_type": "markdown", "id": cid(), "metadata": {}, "source": list(lines)}

cells = []

# ── Header ────────────────────────────────────────────────────────────────────
cells.append(md(
    "# Phase 5: AlphaMissense Comparison\n",
    "Compare the project's PAE-weighted GATv2 model against AlphaMissense "
    "(Cheng et al., 2023, *Science*) on the same held-out test set.\n",
    "\n",
    "**Why this comparison matters:** AlphaMissense is the current state-of-the-art "
    "published tool for missense variant pathogenicity prediction. Benchmarking "
    "against it on the same test variants answers whether a GNN trained on ClinVar "
    "with explicit 3D structural graphs can match a much larger pre-trained model "
    "that uses evolutionary and structural signals from AlphaFold.\n",
    "\n",
    "**Inputs:**\n",
    "- `clinvar_graph_ready.csv` — the same dataset used in Phases 1–4\n",
    "- `models/gatv2_pae_weighted_final_best.pt` — tuned PAE-weighted GATv2 (Phase 4 Section 11)\n",
    "- `models/graphsage_baseline_best.pt` — GraphSAGE baseline (Phase 4 Section 7, best performer)\n",
    "- `best_config.json` — winning hyperparameter config from Phase 4\n",
    "- AlphaMissense aa-substitution scores — downloaded from Zenodo (DOI 10.5281/zenodo.8208688), "
    "stream-filtered to test proteins, cached as `alphamissense_test_scores.csv`\n",
    "\n",
    "**Outputs:**\n",
    "- `alphamissense_test_scores.csv` — filtered AlphaMissense scores for test proteins (cached)\n",
    "- `phase5_results.csv` — head-to-head metrics table on the matched subset\n",
    "- `phase5_significance_tests.csv` — paired bootstrap and DeLong significance tests\n",
    "- `figures/phase5_comparison.png` — AUPRC/AUROC comparison bar chart\n",
))

# ── Imports ────────────────────────────────────────────────────────────────────
cells.append(code(
    "import os, json, gzip, random\n",
    "import requests\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import matplotlib.pyplot as plt\n",
    "from tqdm.notebook import tqdm\n",
    "from torch.utils.data import Sampler\n",
    "from torch_geometric.loader import DataLoader\n",
    "from sklearn.metrics import (average_precision_score, roc_auc_score,\n",
    "                              precision_score, recall_score, f1_score)\n",
    "from scipy import stats\n",
    "\n",
    "from missense_dataset import (\n",
    "    MissenseVariantGraphDataset, NODE_FEATURE_DIM, ESM2_EMBEDDING_DIM, N_AA\n",
    ")\n",
    "from models import GATv2Classifier, GraphSAGEClassifier\n",
    "\n",
    "DEVICE        = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
    "GRAPH_DIR     = \"protein_graphs\"\n",
    "FIGURES_DIR   = \"figures\"\n",
    "AM_CACHE_PATH = \"alphamissense_test_scores.csv\"\n",
    "AM_FULL_PATH  = \"AlphaMissense_aa_substitutions.tsv.gz\"\n",
    "AM_URL        = \"https://zenodo.org/records/8208688/files/AlphaMissense_aa_substitutions.tsv.gz\"\n",
    "\n",
    "os.makedirs(FIGURES_DIR, exist_ok=True)\n",
    "torch.manual_seed(42)\n",
    "np.random.seed(42)\n",
    "print(f\"Device: {DEVICE}\")\n",
    "if DEVICE == \"cuda\":\n",
    "    print(f\"GPU: {torch.cuda.get_device_name(0)}\")\n",
))

# ── Section 1 ──────────────────────────────────────────────────────────────────
cells.append(md(
    "## 1. Reproduce the Phase 4 Test Split\n",
    "Identical protein-level split logic (seed=42) as Phase 4 — guarantees we evaluate "
    "on exactly the same 6,771 variants the GNN was never trained on.",
))
cells.append(code(
    "df = pd.read_csv(\"clinvar_graph_ready.csv\")\n",
    "print(f\"Loaded {len(df):,} variants, {df['UniProt_ID'].nunique():,} proteins\")\n",
    "\n",
    "def make_protein_split(df, seed, train_frac=0.70, val_frac=0.15):\n",
    "    rng = np.random.RandomState(seed)\n",
    "    unique_proteins = df[\"UniProt_ID\"].unique().copy()\n",
    "    rng.shuffle(unique_proteins)\n",
    "    n = len(unique_proteins)\n",
    "    n_train, n_val = int(n * train_frac), int(n * val_frac)\n",
    "    train_p = set(unique_proteins[:n_train])\n",
    "    val_p   = set(unique_proteins[n_train:n_train + n_val])\n",
    "    test_p  = set(unique_proteins[n_train + n_val:])\n",
    "    assert train_p.isdisjoint(val_p) and train_p.isdisjoint(test_p) and val_p.isdisjoint(test_p)\n",
    "    return (df[df[\"UniProt_ID\"].isin(p)].reset_index(drop=True)\n",
    "            for p in (train_p, val_p, test_p))\n",
    "\n",
    "_, _, test_df = make_protein_split(df, seed=42)\n",
    "test_proteins = set(test_df[\"UniProt_ID\"].unique())\n",
    "print(f\"Test set : {len(test_df):,} variants, {len(test_proteins):,} proteins\")\n",
    "print(f\"Pathogenic fraction: {test_df['Target'].mean():.3f}\")\n",
))

# ── Section 2 ──────────────────────────────────────────────────────────────────
cells.append(md(
    "## 2. Load GNN Models and Compute Test Predictions\n",
    "We load two models from their Phase 4 checkpoints:\n",
    "- **GATv2 (PAE-weighted, tuned)** — the project's proposed model (Section 11 checkpoint)\n",
    "- **GraphSAGE (baseline)** — the best-performing Phase 4 model (Section 7 checkpoint), "
    "included to give AlphaMissense a fair comparison against the strongest GNN\n",
    "\n",
    "Both are evaluated on the test set using the same NodeBudgetBatchSampler as Phase 4.\n",
    "\n",
    "Predictions are cached to `phase5_gnn_predictions.csv` after inference — "
    "if that file already exists the models are not reloaded and inference is skipped entirely.",
))
cells.append(code(
    "PRED_CACHE = \"phase5_gnn_predictions.csv\"\n",
    "\n",
    "# ── NodeBudgetBatchSampler (same as Phase 4) ──────────────────────────────\n",
    "class NodeBudgetBatchSampler(Sampler):\n",
    "    def __init__(self, node_counts, max_nodes_per_batch=20000, shuffle=False, seed=0):\n",
    "        self.node_counts = node_counts\n",
    "        self.max_nodes_per_batch = max_nodes_per_batch\n",
    "        self.shuffle = shuffle; self.seed = seed; self._epoch = 0\n",
    "\n",
    "    def __iter__(self):\n",
    "        indices = list(range(len(self.node_counts)))\n",
    "        if self.shuffle:\n",
    "            r = random.Random(self.seed + self._epoch); r.shuffle(indices); self._epoch += 1\n",
    "        batch, current = [], 0\n",
    "        for idx in indices:\n",
    "            n = self.node_counts[idx]\n",
    "            if batch and current + n > self.max_nodes_per_batch:\n",
    "                yield batch; batch, current = [], 0\n",
    "            batch.append(idx); current += n\n",
    "        if batch:\n",
    "            yield batch\n",
    "\n",
    "    def __len__(self):\n",
    "        return max(1, sum(self.node_counts) // self.max_nodes_per_batch)\n",
    "\n",
    "\n",
    "def get_node_counts(variant_df, graph_dir):\n",
    "    cache, counts = {}, []\n",
    "    for uid in variant_df[\"UniProt_ID\"]:\n",
    "        if uid not in cache:\n",
    "            g = torch.load(os.path.join(graph_dir, f\"{uid}.pt\"), weights_only=False)\n",
    "            cache[uid] = g.num_residues\n",
    "        counts.append(cache[uid])\n",
    "    return counts\n",
    "\n",
    "\n",
    "def evaluate_model(model, test_df, graph_dir, device, desc=\"inference\"):\n",
    "    \"\"\"Returns (labels, probabilities) for all test variants.\"\"\"\n",
    "    ds = MissenseVariantGraphDataset(test_df, graph_dir)\n",
    "    nc = get_node_counts(test_df, graph_dir)\n",
    "    loader = DataLoader(ds, batch_sampler=NodeBudgetBatchSampler(nc, shuffle=False))\n",
    "    model.eval()\n",
    "    logits, labels = [], []\n",
    "    with torch.no_grad():\n",
    "        for batch in tqdm(loader, desc=desc):\n",
    "            batch = batch.to(device)\n",
    "            logits.extend(model(batch).cpu().numpy().tolist())\n",
    "            labels.extend(batch.y.cpu().numpy().tolist())\n",
    "    probs = torch.sigmoid(torch.tensor(logits)).numpy()\n",
    "    return np.array(labels), probs\n",
    "\n",
    "\n",
    "if os.path.exists(PRED_CACHE):\n",
    "    pred_cache = pd.read_csv(PRED_CACHE)\n",
    "    test_df = test_df.merge(pred_cache[[\"VariationID\", \"gatv2_prob\", \"sage_prob\", \"true_label\"]],\n",
    "                             on=\"VariationID\", how=\"left\")\n",
    "    test_labels  = test_df[\"true_label\"].values.astype(int)\n",
    "    gatv2_probs  = test_df[\"gatv2_prob\"].values\n",
    "    sage_probs   = test_df[\"sage_prob\"].values\n",
    "    print(f\"Loaded cached GNN predictions from {PRED_CACHE} -- skipping inference.\")\n",
    "    print(f\"  GATv2     AUPRC={average_precision_score(test_labels, gatv2_probs):.4f}\")\n",
    "    print(f\"  GraphSAGE AUPRC={average_precision_score(test_labels, sage_probs):.4f}\")\n",
    "else:\n",
    "    # ── Load best_config ──────────────────────────────────────────────────────\n",
    "    with open(\"best_config.json\") as f:\n",
    "        best_config = json.load(f)\n",
    "    print(f\"best_config: {best_config['name']}  hidden_dim={int(best_config['hidden_dim'])}\")\n",
    "\n",
    "    # ── GATv2 PAE-weighted (tuned) ────────────────────────────────────────────\n",
    "    model_gatv2 = GATv2Classifier(in_dim=NODE_FEATURE_DIM, use_edge_attr=True,\n",
    "                                   hidden_dim=int(best_config[\"hidden_dim\"]),\n",
    "                                   dropout=best_config[\"dropout\"])\n",
    "    model_gatv2.load_state_dict(\n",
    "        torch.load(\"models/gatv2_pae_weighted_final_best.pt\", weights_only=True))\n",
    "    model_gatv2.to(DEVICE)\n",
    "    print(\"Loaded GATv2 (PAE-weighted, tuned)\")\n",
    "    test_labels, gatv2_probs = evaluate_model(\n",
    "        model_gatv2, test_df, GRAPH_DIR, DEVICE, \"GATv2 inference\")\n",
    "    print(f\"GATv2  -> AUPRC={average_precision_score(test_labels, gatv2_probs):.4f}  \"\n",
    "          f\"AUROC={roc_auc_score(test_labels, gatv2_probs):.4f}\")\n",
    "    del model_gatv2\n",
    "    if DEVICE == \"cuda\": torch.cuda.empty_cache()\n",
    "\n",
    "    # ── GraphSAGE (baseline) ──────────────────────────────────────────────────\n",
    "    model_sage = GraphSAGEClassifier(in_dim=NODE_FEATURE_DIM)\n",
    "    model_sage.load_state_dict(\n",
    "        torch.load(\"models/graphsage_baseline_best.pt\", weights_only=True))\n",
    "    model_sage.to(DEVICE)\n",
    "    print(\"Loaded GraphSAGE (baseline)\")\n",
    "    _, sage_probs = evaluate_model(\n",
    "        model_sage, test_df, GRAPH_DIR, DEVICE, \"GraphSAGE inference\")\n",
    "    print(f\"GraphSAGE -> AUPRC={average_precision_score(test_labels, sage_probs):.4f}  \"\n",
    "          f\"AUROC={roc_auc_score(test_labels, sage_probs):.4f}\")\n",
    "    del model_sage\n",
    "    if DEVICE == \"cuda\": torch.cuda.empty_cache()\n",
    "\n",
    "    # ── Cache predictions to disk ─────────────────────────────────────────────\n",
    "    test_df = test_df.copy()\n",
    "    test_df[\"true_label\"] = test_labels\n",
    "    test_df[\"gatv2_prob\"] = gatv2_probs\n",
    "    test_df[\"sage_prob\"]  = sage_probs\n",
    "    test_df[[\"VariationID\", \"UniProt_ID\", \"Pos\", \"Ref_AA\", \"Alt_AA_1\",\n",
    "              \"Target\", \"true_label\", \"gatv2_prob\", \"sage_prob\"]].to_csv(\n",
    "        PRED_CACHE, index=False)\n",
    "    print(f\"Saved GNN predictions to {PRED_CACHE}  \"\n",
    "          f\"(delete this file to force re-inference)\")\n",
))

# ── Section 3 ──────────────────────────────────────────────────────────────────
cells.append(md(
    "## 3. Download and Cache AlphaMissense Scores\n",
    "AlphaMissense provides a protein-level aa-substitution file (~390 MB compressed) "
    "on Zenodo (DOI 10.5281/zenodo.8208688). We stream-read it, keeping only rows "
    "whose `uniprot_id` is in our test set (~991 proteins), and cache the result as "
    "`alphamissense_test_scores.csv` so the download never repeats.\n",
    "\n",
    "**If the download fails** (network error, Zenodo outage): manually download "
    "`AlphaMissense_aa_substitutions.tsv.gz` from "
    "https://zenodo.org/records/8208688 and place it in the project root. "
    "Re-running this cell will then filter it locally without re-downloading.",
))
cells.append(code(
    "def load_am_scores(test_proteins, cache_path, full_path, url):\n",
    "    if os.path.exists(cache_path):\n",
    "        am = pd.read_csv(cache_path)\n",
    "        print(f\"Loaded cached AlphaMissense scores from {cache_path} ({len(am):,} rows)\")\n",
    "        return am\n",
    "\n",
    "    # Download if the full file isn't on disk yet\n",
    "    if not os.path.exists(full_path):\n",
    "        print(f\"Downloading AlphaMissense aa-substitutions file (~390 MB compressed)...\")\n",
    "        print(f\"Source: {url}\")\n",
    "        with requests.get(url, stream=True, timeout=120) as r:\n",
    "            r.raise_for_status()\n",
    "            total = int(r.headers.get(\"content-length\", 0))\n",
    "            downloaded = 0\n",
    "            with open(full_path, \"wb\") as f:\n",
    "                for chunk in r.iter_content(chunk_size=65536):\n",
    "                    f.write(chunk)\n",
    "                    downloaded += len(chunk)\n",
    "                    if total:\n",
    "                        print(f\"\\r  {downloaded/1e6:.1f} / {total/1e6:.1f} MB\", end=\"\", flush=True)\n",
    "        print(f\"\\nSaved to {full_path}\")\n",
    "\n",
    "    # Stream-filter: keep only rows for our test proteins\n",
    "    print(f\"Filtering to {len(test_proteins):,} test proteins (stream-reading {full_path})...\")\n",
    "    rows = []\n",
    "    header = None\n",
    "    with gzip.open(full_path, \"rt\", encoding=\"utf-8\") as fh:\n",
    "        for line in tqdm(fh, desc=\"Reading AlphaMissense\", unit=\" lines\", mininterval=2):\n",
    "            if line.startswith(\"#\"):\n",
    "                # The actual column header line starts with '#' in this file\n",
    "                stripped = line.lstrip(\"#\").strip()\n",
    "                if \"uniprot_id\" in stripped or \"protein_variant\" in stripped:\n",
    "                    header = stripped.split(\"\\t\")\n",
    "                continue\n",
    "            if header is None:\n",
    "                header = line.strip().split(\"\\t\")\n",
    "                continue\n",
    "            parts = line.strip().split(\"\\t\")\n",
    "            if parts[0] in test_proteins:\n",
    "                rows.append(dict(zip(header, parts)))\n",
    "\n",
    "    am = pd.DataFrame(rows)\n",
    "    am[\"am_pathogenicity\"] = am[\"am_pathogenicity\"].astype(float)\n",
    "    am.to_csv(cache_path, index=False)\n",
    "    print(f\"Filtered to {len(am):,} rows — saved to {cache_path}\")\n",
    "    return am\n",
    "\n",
    "\n",
    "am_scores = load_am_scores(test_proteins, AM_CACHE_PATH, AM_FULL_PATH, AM_URL)\n",
    "print(f\"\\nColumns : {list(am_scores.columns)}\")\n",
    "print(f\"Sample  :\\n{am_scores.head(3).to_string(index=False)}\")\n",
    "print(f\"\\nClass distribution:\\n{am_scores['am_class'].value_counts().to_string()}\")\n",
))

# ── Section 4 ──────────────────────────────────────────────────────────────────
cells.append(md(
    "## 4. Join AlphaMissense Scores to the Test Set\n",
    "AlphaMissense's `protein_variant` format is `{WT_1letter}{position}{ALT_1letter}` "
    "(e.g. `G281V`). Our `Ref_AA` column uses 3-letter codes, so we convert it first. "
    "We then join on `(UniProt_ID, variant_key)` and report coverage.\n",
    "\n",
    "**Coverage < 100%** is expected: AlphaMissense covers ~19,000 human proteins, but "
    "some ClinVar entries may reference non-canonical isoforms or use positions that "
    "don't align to the UniProt canonical sequence. The comparison runs on the matched "
    "subset only, and we verify that its class balance is similar to the full test set "
    "so the matched subset is representative.",
))
cells.append(code(
    "AA3TO1 = {\n",
    "    \"Ala\": \"A\", \"Arg\": \"R\", \"Asn\": \"N\", \"Asp\": \"D\", \"Cys\": \"C\",\n",
    "    \"Gln\": \"Q\", \"Glu\": \"E\", \"Gly\": \"G\", \"His\": \"H\", \"Ile\": \"I\",\n",
    "    \"Leu\": \"L\", \"Lys\": \"K\", \"Met\": \"M\", \"Phe\": \"F\", \"Pro\": \"P\",\n",
    "    \"Ser\": \"S\", \"Thr\": \"T\", \"Trp\": \"W\", \"Tyr\": \"Y\", \"Val\": \"V\",\n",
    "}\n",
    "\n",
    "test_df[\"ref_1\"] = test_df[\"Ref_AA\"].map(AA3TO1)\n",
    "unmapped = test_df[\"ref_1\"].isna().sum()\n",
    "if unmapped:\n",
    "    print(f\"WARNING: {unmapped} variants with unrecognised Ref_AA — will be excluded from join:\")\n",
    "    print(test_df[test_df[\"ref_1\"].isna()][\"Ref_AA\"].value_counts())\n",
    "\n",
    "test_df[\"variant_key\"] = test_df[\"ref_1\"] + test_df[\"Pos\"].astype(str) + test_df[\"Alt_AA_1\"]\n",
    "\n",
    "am_clean = am_scores.rename(columns={\n",
    "    \"uniprot_id\": \"UniProt_ID\", \"protein_variant\": \"variant_key\"\n",
    "})[[\"UniProt_ID\", \"variant_key\", \"am_pathogenicity\", \"am_class\"]]\n",
    "\n",
    "merged = test_df.merge(am_clean, on=[\"UniProt_ID\", \"variant_key\"], how=\"inner\")\n",
    "\n",
    "coverage = len(merged) / len(test_df) * 100\n",
    "print(f\"Coverage : {len(merged):,} / {len(test_df):,} test variants matched ({coverage:.1f}%)\")\n",
    "print(f\"Pathogenic fraction — full test set : {test_df['Target'].mean():.3f}\")\n",
    "print(f\"Pathogenic fraction — matched subset: {merged['Target'].mean():.3f}\")\n",
    "print(f\"\\nAlphaMissense class distribution in matched subset:\")\n",
    "print(merged[\"am_class\"].value_counts().to_string())\n",
))

# ── Section 5 ──────────────────────────────────────────────────────────────────
cells.append(md(
    "## 5. Head-to-Head Metrics on the Matched Subset\n",
    "All three models are re-evaluated on exactly the same matched variants so the "
    "comparison is fair. AUPRC is the primary metric (same rationale as Phase 4: "
    "class imbalance makes AUPRC more informative than accuracy or AUROC alone). "
    "Classification metrics (Precision/Recall/F1) use a 0.5 threshold for the GNN "
    "models and AlphaMissense's published threshold of **0.564** (the midpoint of "
    "its ambiguous band), noted separately.",
))
cells.append(code(
    "labels  = merged[\"true_label\"].values.astype(int)\n",
    "g_probs = merged[\"gatv2_prob\"].values\n",
    "s_probs = merged[\"sage_prob\"].values\n",
    "a_probs = merged[\"am_pathogenicity\"].values\n",
    "\n",
    "AM_THRESHOLD = 0.564  # AlphaMissense published likely-pathogenic threshold\n",
    "\n",
    "def metrics_row(name, probs, labels, threshold=0.5):\n",
    "    preds = (probs >= threshold).astype(int)\n",
    "    return {\n",
    "        \"Model\": name,\n",
    "        \"AUPRC\":     round(average_precision_score(labels, probs), 4),\n",
    "        \"AUROC\":     round(roc_auc_score(labels, probs), 4),\n",
    "        \"Precision\": round(precision_score(labels, preds, zero_division=0), 4),\n",
    "        \"Recall\":    round(recall_score(labels, preds, zero_division=0), 4),\n",
    "        \"F1\":        round(f1_score(labels, preds, zero_division=0), 4),\n",
    "        \"Threshold\": threshold,\n",
    "    }\n",
    "\n",
    "results = [\n",
    "    metrics_row(\"GATv2 (PAE-weighted, tuned)\", g_probs, labels, 0.5),\n",
    "    metrics_row(\"GraphSAGE (baseline)\",        s_probs, labels, 0.5),\n",
    "    metrics_row(\"AlphaMissense\",               a_probs, labels, AM_THRESHOLD),\n",
    "]\n",
    "results_df = pd.DataFrame(results).sort_values(\"AUPRC\", ascending=False)\n",
    "results_df.to_csv(\"phase5_results.csv\", index=False)\n",
    "print(\"=== Results on matched subset ===\")\n",
    "print(results_df.to_string(index=False))\n",
    "print(f\"\\nn_matched = {len(merged):,} variants\")\n",
))

# ── Section 6 ──────────────────────────────────────────────────────────────────
cells.append(md(
    "## 6. Statistical Significance\n",
    "Same paired tests as Phase 4 Sections 9/13: paired bootstrap on AUPRC difference "
    "(2,000 resamples, threshold-free) and DeLong's test on AUROC. "
    "Key comparisons: GATv2 vs AlphaMissense, and GraphSAGE vs AlphaMissense.",
))
cells.append(code(
    "def paired_bootstrap_auprc(pa, pb, labels, n_boot=2000, seed=0):\n",
    "    rng = np.random.RandomState(seed)\n",
    "    pa, pb, labels = np.asarray(pa), np.asarray(pb), np.asarray(labels)\n",
    "    diffs = np.array([\n",
    "        average_precision_score(labels[idx := rng.randint(0, len(labels), len(labels))], pa[idx])\n",
    "        - average_precision_score(labels[idx], pb[idx])\n",
    "        for _ in range(n_boot)\n",
    "    ])\n",
    "    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])\n",
    "    p = 2 * min((diffs > 0).mean(), (diffs < 0).mean())\n",
    "    return {\"mean_diff\": float(diffs.mean()), \"ci_low\": float(ci_lo),\n",
    "            \"ci_high\": float(ci_hi), \"p_value\": float(p)}\n",
    "\n",
    "\n",
    "def compute_midrank(x):\n",
    "    J = np.argsort(x); Z = x[J]; N = len(x)\n",
    "    T = np.zeros(N, dtype=float)\n",
    "    i = 0\n",
    "    while i < N:\n",
    "        j = i\n",
    "        while j < N and Z[j] == Z[i]: j += 1\n",
    "        T[i:j] = 0.5 * (i + j - 1) + 1; i = j\n",
    "    T2 = np.empty(N, dtype=float); T2[J] = T\n",
    "    return T2\n",
    "\n",
    "\n",
    "def delong_test(pa, pb, labels):\n",
    "    labels = np.asarray(labels)\n",
    "    order = np.argsort(-labels)\n",
    "    m = int((labels == 1).sum()); n = int((labels == 0).sum())\n",
    "    preds = [np.asarray(pa), np.asarray(pb)]\n",
    "    k = 2\n",
    "    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])\n",
    "    for r in range(k):\n",
    "        po = preds[r][order]\n",
    "        tx[r] = compute_midrank(po[:m])\n",
    "        ty[r] = compute_midrank(po[m:])\n",
    "        tz[r] = compute_midrank(po)\n",
    "    aucs = tz[:, :m].sum(1) / m / n - float(m + 1.0) / 2.0 / n\n",
    "    v01 = (tz[:, :m] - tx) / n\n",
    "    v10 = 1.0 - (tz[:, m:] - ty) / m\n",
    "    cov = np.cov(v01) / m + np.cov(v10) / n\n",
    "    assert abs(aucs[0] - roc_auc_score(labels, pa)) < 1e-6\n",
    "    assert abs(aucs[1] - roc_auc_score(labels, pb)) < 1e-6\n",
    "    l = np.array([1, -1])\n",
    "    z = l.dot(aucs) / np.sqrt(l.dot(cov).dot(l))\n",
    "    p = 2 * (1 - stats.norm.cdf(abs(z)))\n",
    "    return {\"auc_a\": float(aucs[0]), \"auc_b\": float(aucs[1]), \"z\": float(z), \"p_value\": float(p)}\n",
    "\n",
    "\n",
    "def run_comparison(name_a, pa, name_b, pb, labels):\n",
    "    bs = paired_bootstrap_auprc(pa, pb, labels)\n",
    "    dl = delong_test(pa, pb, labels)\n",
    "    sig = bs[\"p_value\"] < 0.05 or dl[\"p_value\"] < 0.05\n",
    "    print(f\"\\n=== {name_a}  vs.  {name_b} ===\")\n",
    "    print(f\"  Bootstrap (AUPRC): diff={bs['mean_diff']:+.4f}  \"\n",
    "          f\"95% CI=[{bs['ci_low']:+.4f}, {bs['ci_high']:+.4f}]  p={bs['p_value']:.4f}\")\n",
    "    print(f\"  DeLong   (AUROC) : z={dl['z']:.3f}  p={dl['p_value']:.4f}\")\n",
    "    verdict = \"STATISTICALLY SIGNIFICANT\" if sig else \"NOT statistically significant\"\n",
    "    print(f\"  -> {verdict} (alpha=0.05)\")\n",
    "    return {\"comparison\": f\"{name_a} vs {name_b}\",\n",
    "            \"bootstrap_diff\": bs[\"mean_diff\"], \"ci_low\": bs[\"ci_low\"],\n",
    "            \"ci_high\": bs[\"ci_high\"], \"bootstrap_p\": bs[\"p_value\"],\n",
    "            \"delong_z\": dl[\"z\"], \"delong_p\": dl[\"p_value\"], \"significant\": sig}\n",
    "\n",
    "\n",
    "sig_results = []\n",
    "sig_results.append(run_comparison(\"GATv2 (PAE-weighted, tuned)\", g_probs,\n",
    "                                   \"AlphaMissense\", a_probs, labels))\n",
    "sig_results.append(run_comparison(\"GraphSAGE (baseline)\", s_probs,\n",
    "                                   \"AlphaMissense\", a_probs, labels))\n",
    "sig_results.append(run_comparison(\"GATv2 (PAE-weighted, tuned)\", g_probs,\n",
    "                                   \"GraphSAGE (baseline)\", s_probs, labels))\n",
    "\n",
    "pd.DataFrame(sig_results).to_csv(\"phase5_significance_tests.csv\", index=False)\n",
    "print(\"\\nSaved phase5_significance_tests.csv\")\n",
))

# ── Section 7 ──────────────────────────────────────────────────────────────────
cells.append(md("## 7. Visualisation"))
cells.append(code(
    "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n",
    "\n",
    "model_names  = [\"GATv2\\n(PAE-weighted,\\ntuned)\", \"GraphSAGE\\n(baseline)\", \"AlphaMissense\"]\n",
    "auprc_vals   = [results_df.set_index(\"Model\").loc[n, \"AUPRC\"]\n",
    "                 if n in results_df[\"Model\"].values else None\n",
    "                 for n in [\"GATv2 (PAE-weighted, tuned)\", \"GraphSAGE (baseline)\", \"AlphaMissense\"]]\n",
    "auroc_vals   = [results_df.set_index(\"Model\").loc[n, \"AUROC\"]\n",
    "                 if n in results_df[\"Model\"].values else None\n",
    "                 for n in [\"GATv2 (PAE-weighted, tuned)\", \"GraphSAGE (baseline)\", \"AlphaMissense\"]]\n",
    "colors = [\"#2196F3\", \"#4CAF50\", \"#FF5722\"]\n",
    "\n",
    "for ax, vals, metric in zip(axes, [auprc_vals, auroc_vals], [\"AUPRC\", \"AUROC\"]):\n",
    "    bars = ax.bar(model_names, vals, color=colors, width=0.5, edgecolor=\"white\")\n",
    "    lo = min(v for v in vals if v) - 0.02\n",
    "    ax.set_ylim(max(0, lo), min(1.0, max(v for v in vals if v) + 0.03))\n",
    "    ax.set_ylabel(metric)\n",
    "    ax.set_title(f\"{metric} — matched subset (n={len(merged):,})\")\n",
    "    for bar, val in zip(bars, vals):\n",
    "        if val is not None:\n",
    "            ax.text(bar.get_x() + bar.get_width() / 2,\n",
    "                    bar.get_height() + 0.001, f\"{val:.4f}\",\n",
    "                    ha=\"center\", va=\"bottom\", fontsize=9)\n",
    "\n",
    "plt.suptitle(\"Phase 5: GNN vs AlphaMissense on matched test variants\", fontsize=12)\n",
    "plt.tight_layout()\n",
    "fig.savefig(os.path.join(FIGURES_DIR, \"phase5_comparison.png\"), dpi=200, bbox_inches=\"tight\")\n",
    "plt.show()\n",
    "plt.close(fig)\n",
))

# ── Section 8 ──────────────────────────────────────────────────────────────────
cells.append(md(
    "## 8. Methods Note (for Paper)\n",
    "> AlphaMissense (Cheng et al., 2023) precomputed pathogenicity scores were downloaded "
    "from Zenodo (DOI 10.5281/zenodo.8208688) using the `AlphaMissense_aa_substitutions.tsv.gz` "
    "file (protein-level UniProt-indexed format). Scores were joined to the Phase 4 test set "
    "by UniProt accession and amino-acid substitution notation (`{WT}{position}{ALT}` in "
    "one-letter code). Variants without a matching AlphaMissense score were excluded from "
    "the comparison; coverage and class-balance representativeness of the matched subset "
    "are reported in Section 4. All metrics (AUPRC, AUROC, Precision, Recall, F1) were "
    "computed on the same matched variants for all models. For classification metrics, "
    "AlphaMissense's published likely-pathogenic threshold of 0.564 was used; GNN models "
    "used 0.5. Significance testing used the same paired bootstrap (2,000 resamples) on "
    "the AUPRC difference and DeLong's paired AUROC test as Phase 4.",
))

# ── Write notebook ─────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "cells": cells
}

out = "phase5_alphamissense_comparison.ipynb"
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Written {len(cells)} cells to {out}")
