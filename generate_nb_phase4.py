import json

def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": list(lines)}

def code(*lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": list(lines)}

cells = []

# ============================================================ 0. Overview
cells.append(md(
    "# Phase 4: Model Training — PAE-Weighted GATv2 vs. Baselines\n",
    "This notebook trains the project's core model — a Graph Attention Network (GATv2) whose edges are weighted by "
    "AlphaFold's Predicted Aligned Error — and compares it against four baselines to test whether each piece of the "
    "architecture is actually earning its place.\n\n"
    "**Inputs:** `clinvar_graph_ready.csv`, `protein_graphs/*.pt` (Phase 3)\n\n"
    "**Models trained:**\n"
    "1. **GATv2 (PAE-weighted)** — the project's proposed model; edge attention is conditioned on `[distance, "
    "PAE-confidence]`.\n"
    "2. **GATv2 (unweighted ablation)** — identical architecture, but edge features are withheld, so it only sees "
    "binary contact structure (the gMVP-style baseline this project is positioned against).\n"
    "3. **GCN (plain baseline)** — a simpler, non-attention graph convolution, also unweighted.\n"
    "4. **GraphSAGE (additional baseline)** — a different, widely-used non-attention GNN (mean-aggregation over "
    "neighbors), broadening architecture coverage beyond a single comparison point.\n"
    "5. **XGBoost (flat, structure-free baseline)** — sees only the variant's own ESM-2 embedding and amino-acid "
    "identities, with no graph/structure at all.\n\n"
    "Comparing all five answers two separate questions: *does structure help at all* (GNNs vs. XGBoost), and *does "
    "PAE-weighting help beyond structure alone* (model 1 vs. models 2-4) — which is this project's specific claim.\n\n"
    "**Before running:** restart the kernel and close other heavy applications first. We measured a real, one-time "
    "host-RAM spike (~5-6 GB, separate from GPU memory) tied to the first model's training in this environment - not "
    "a leak (it partially releases afterward), but worth having headroom for given this machine's 15.8 GB total RAM. "
    "We also measured **one full training epoch at ~31 minutes** on this GPU (RTX 4050, 6.4 GB) over the full "
    "34,732-variant training split - this notebook is compute-heavy by design (hyperparameter search + a final "
    "tuned run + 3-fold cross-validation on top of the initial comparison) and is intended to run unattended over "
    "multiple sessions, not in one sitting.\n\n"
    "**Why this notebook does more than report one results table.** An initial pass (Section 8) showed all models "
    "performing within ~1-2 points of each other, with the GNNs overfitting within the first few epochs. Rather than "
    "stopping there, this notebook: checks *why* (Section 2's PAE-signal diagnostic; Section 8's training curves), "
    "tests whether the differences are even statistically real (Section 9) before concluding anything, runs a "
    "constrained hyperparameter search and a final tuned training pass that directly targets the overfitting "
    "(Sections 10-11), re-tests significance on the tuned models (Section 13), and finally checks how much the "
    "tuned model's performance varies across different train/test splits via 3-fold cross-validation (Section 14) - "
    "a single test-set number alone cannot answer that.\n\n"
    "**Outputs:**\n"
    "- `models/{model_name}_best.pt` — best checkpoint per model, reloadable via the shared `models.py` module\n"
    "- `figures/*.png` — every figure in this notebook, saved at publication resolution\n"
    "- `phase4_results_initial.csv` / `phase4_results_final.csv` — comparison tables (AUPRC, AUROC, Precision, "
    "Recall, F1) before and after tuning\n"
    "- `phase4_hyperparameter_search.csv` — every search trial's configuration and validation AUPRC\n"
    "- `phase4_cv_results.csv` — per-fold and mean±std test metrics from the 3-fold cross-validation\n"
    "- `phase4_training_history_{model_name}.csv` — per-epoch train/val loss and AUPRC, for overfitting diagnosis\n"
    "- `phase4_significance_tests_*.csv` — paired statistical test results for every model comparison that matters "
    "to this project's claims\n\n"
    "**Key sources:**\n"
    "- Brody, S., Alon, U., & Yahav, E. (2022). *How Attentive Are Graph Attention Networks?* ICLR 2022. "
    "[arXiv:2105.14491](https://arxiv.org/abs/2105.14491) — defines GATv2.\n"
    "- Hamilton, W., Ying, Z., & Leskovec, J. (2017). *Inductive Representation Learning on Large Graphs.* "
    "NeurIPS 2017. [arXiv:1706.02216](https://arxiv.org/abs/1706.02216) — defines GraphSAGE.\n"
    "- Davis, J. & Goadrich, M. (2006). *The Relationship Between Precision-Recall and ROC Curves.* ICML 2006 — "
    "motivates AUPRC as the primary metric under class imbalance.\n"
    "- Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD 2016. "
    "[doi:10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785) — the flat-feature baseline.\n"
    "- DeLong, E.R., DeLong, D.M., & Clarke-Pearson, D.L. (1988). *Comparing the areas under two or more correlated "
    "receiver operating characteristic curves.* Biometrics, 44(3), 837-845 — the paired AUROC significance test.\n"
    "- McNemar, Q. (1947). *Note on the sampling error of the difference between correlated proportions or "
    "percentages.* Psychometrika, 12(2), 153-157 — the paired classification-agreement test.\n"
    "- Kohavi, R. (1995). *A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection.* "
    "IJCAI 1995 — motivates k-fold cross-validation for robustness estimation (Section 14)."
))

cells.append(code(
    "import os\n",
    "import json\n",
    "import random\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import matplotlib.pyplot as plt\n",
    "from scipy import stats\n",
    "from torch.utils.data import Sampler\n",
    "from torch_geometric.loader import DataLoader\n",
    "from torch_geometric.data import Batch\n",
    "from sklearn.metrics import (average_precision_score, roc_auc_score, confusion_matrix,\n",
    "                              precision_score, recall_score, f1_score)\n",
    "from statsmodels.stats.contingency_tables import mcnemar\n",
    "from tqdm.notebook import tqdm\n",
    "\n",
    "from missense_dataset import (\n",
    "    MissenseVariantGraphDataset, NODE_FEATURE_DIM, ESM2_EMBEDDING_DIM, N_AA, AA_1TO_ONEHOT_IDX\n",
    ")\n",
    "from models import GATv2Classifier, GCNClassifier, GraphSAGEClassifier\n",
    "\n",
    "DEVICE = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
    "print(f\"Using device: {DEVICE}\")\n",
    "if DEVICE == \"cuda\":\n",
    "    print(f\"GPU: {torch.cuda.get_device_name(0)}\")\n",
    "    print(f\"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB\")\n",
    "\n",
    "GRAPH_DIR = \"protein_graphs\"\n",
    "FIGURES_DIR = \"figures\"\n",
    "os.makedirs(\"models\", exist_ok=True)\n",
    "os.makedirs(FIGURES_DIR, exist_ok=True)\n",
    "torch.manual_seed(42)\n",
    "np.random.seed(42)"
))

# ============================================================ 1. Load data
cells.append(md(
    "## 1. Load Phase 3 Output"
))

cells.append(code(
    "df = pd.read_csv(\"clinvar_graph_ready.csv\")\n",
    "print(f\"Total variants : {len(df):,}\")\n",
    "print(f\"Unique proteins: {df['UniProt_ID'].nunique():,}\")\n",
    "print(f\"  Pathogenic (1): {(df['Target']==1).sum():,}\")\n",
    "print(f\"  Benign     (0): {(df['Target']==0).sum():,}\")"
))

# ============================================================ 2. PAE diagnostic
cells.append(md(
    "## 2. Diagnostic: Does PAE Actually Carry Enough Signal to Be Useful?\n",
    "Before training anything, it's worth directly checking the assumption the entire PAE-weighting idea rests on: "
    "that PAE varies meaningfully across a protein's edges, giving the attention mechanism something to discriminate "
    "on. If most edges end up with nearly the same weight, there is little for the model to learn from, regardless of "
    "architecture.\n\n"
    "**Why this matters specifically for the transform we used (`1/(1+PAE)`):** AlphaFold's reported PAE rarely goes "
    "below ~1 Å even for the most confident residue pairs, which puts a practical *ceiling* on this transform "
    "(`1/(1+1) = 0.5`) - the weight can get close to its analytic maximum of 1.0 only when PAE approaches 0, which is "
    "uncommon in practice. If a large share of edges sit at or near this ceiling, the transform compresses most of "
    "the dataset into a narrow effective range, leaving only the (smaller) population of long-range, less-confident "
    "contacts to carry meaningful variation."
))

cells.append(code(
    "rng_diag = np.random.RandomState(0)\n",
    "sample_proteins_diag = rng_diag.choice(df[\"UniProt_ID\"].unique(), size=300, replace=False)\n",
    "\n",
    "all_weights = []\n",
    "per_protein_std = []\n",
    "for uid in tqdm(sample_proteins_diag, desc=\"Sampling PAE weights\"):\n",
    "    g = torch.load(os.path.join(GRAPH_DIR, f\"{uid}.pt\"), weights_only=False)\n",
    "    w = g.edge_attr[:, 1].numpy()\n",
    "    all_weights.append(w)\n",
    "    per_protein_std.append(w.std())\n",
    "all_weights = np.concatenate(all_weights)\n",
    "\n",
    "print(f\"Sampled {len(sample_proteins_diag)} proteins, {len(all_weights):,} total edges\\n\")\n",
    "print(f\"Global PAE-weight distribution: min={all_weights.min():.4f} max={all_weights.max():.4f} \"\n",
    "      f\"mean={all_weights.mean():.4f} std={all_weights.std():.4f}\")\n",
    "for p in [1, 5, 25, 50, 75, 95, 99]:\n",
    "    print(f\"  p{p:2d}: {np.percentile(all_weights, p):.4f}\")\n",
    "\n",
    "print(f\"\\nPer-protein std of edge weights: mean={np.mean(per_protein_std):.4f}, \"\n",
    "      f\"median={np.median(per_protein_std):.4f}\")\n",
    "print(\"\\nInterpretation: if the 50th/75th/95th percentiles above are all sitting at the same value, most edges\\n\"\n",
    "      \"are saturated at the practical ceiling of the weight transform, and only a minority of (long-range,\\n\"\n",
    "      \"low-confidence) edges differ from each other - exactly the kind of limited-signal situation that would\\n\"\n",
    "      \"explain a PAE-weighted model failing to outperform an unweighted one, independent of training setup.\")"
))

# ============================================================ 3. Split
cells.append(md(
    "## 3. Protein-Level Train / Validation / Test Split\n",
    "### Why split by protein (UniProt_ID), not by individual variant\n"
    "Many variants on the same protein share the exact same structural graph. If two variants of the same protein "
    "landed in both the training and test sets, the model could effectively 'see' that protein's structure during "
    "training and then be evaluated on a held-out variant of a structure it already learned — inflating test "
    "performance. Splitting at the protein level guarantees no protein's structure appears in more than one split.\n\n"
    "### Why protein (UniProt_ID), specifically, rather than gene symbol\n"
    "We checked empirically: of the 6,597 proteins in this dataset, **15 UniProt accessions are referenced under two "
    "different `GeneSymbol` values** (e.g. `HBA1`/`HBA2`, separate genes encoding an identical protein product). "
    "Splitting by gene symbol could place the *same* protein structure in both splits under two different gene "
    "names. Splitting by `UniProt_ID` closes this gap.\n\n"
    "### A reusable split function\n"
    "Section 14 needs this same splitting logic repeated under different random seeds for cross-validation, so we "
    "define it once as a function (with the disjointness assertion built in as the actual leakage guarantee, not "
    "just a comment) rather than duplicating the logic."
))

cells.append(code(
    "def make_protein_split(df, seed, train_frac=0.70, val_frac=0.15):\n",
    "    \"\"\"Protein-level train/val/test split. Returns three disjoint-by-protein dataframes.\"\"\"\n",
    "    rng = np.random.RandomState(seed)\n",
    "    unique_proteins = df[\"UniProt_ID\"].unique().copy()\n",
    "    rng.shuffle(unique_proteins)\n",
    "    n = len(unique_proteins)\n",
    "    n_train, n_val = int(n * train_frac), int(n * val_frac)\n",
    "    train_p = set(unique_proteins[:n_train])\n",
    "    val_p = set(unique_proteins[n_train:n_train + n_val])\n",
    "    test_p = set(unique_proteins[n_train + n_val:])\n",
    "    assert train_p.isdisjoint(val_p) and train_p.isdisjoint(test_p) and val_p.isdisjoint(test_p), \\\n",
    "        \"Protein split is not leak-free!\"\n",
    "    train_df = df[df[\"UniProt_ID\"].isin(train_p)].reset_index(drop=True)\n",
    "    val_df = df[df[\"UniProt_ID\"].isin(val_p)].reset_index(drop=True)\n",
    "    test_df = df[df[\"UniProt_ID\"].isin(test_p)].reset_index(drop=True)\n",
    "    return train_df, val_df, test_df\n",
    "\n",
    "\n",
    "train_df, val_df, test_df = make_protein_split(df, seed=42)\n",
    "print(\"Protein-level split is leak-free (verified disjoint by assertion above).\")\n",
    "for name, sub in [(\"Train\", train_df), (\"Val\", val_df), (\"Test\", test_df)]:\n",
    "    frac = (sub[\"Target\"] == 1).mean()\n",
    "    print(f\"{name:5s}: {len(sub):6,} variants, {sub['UniProt_ID'].nunique():4,} proteins, \"\n",
    "          f\"{frac:.3f} pathogenic fraction\")"
))

# ============================================================ 4. Node-budget batching
cells.append(md(
    "## 4. Memory-Safe Batching: a Node-Budget Sampler\n",
    "### Why a fixed batch size (e.g. \"16 graphs per batch\") is unsafe here\n"
    "Protein sizes in this dataset range from a handful of residues to 2,699 (Phase 1's cutoff). A fixed *graph-count* "
    "batch size doesn't account for this: we measured a real worst case — 16 of our largest proteins landing in the "
    "same batch — and it peaked at **7.26 GB**, *exceeding* this machine's 6.44 GB of GPU memory.\n\n"
    "### The fix: cap total nodes per batch, not graphs per batch\n"
    "`NodeBudgetBatchSampler` greedily packs variants into a batch until adding the next one would exceed a fixed "
    "**node budget**, bounding peak memory deterministically regardless of shuffle order. Verified under an "
    "adversarial worst-case ordering: peak memory **3.47 GB**."
))

cells.append(code(
    "class NodeBudgetBatchSampler(Sampler):\n",
    "    \"\"\"\n",
    "    Yields batches of dataset indices such that the total node count (sum of protein lengths) in any\n",
    "    one batch never exceeds max_nodes_per_batch - bounding peak GPU memory regardless of which large\n",
    "    proteins happen to land next to each other after shuffling.\n",
    "    \"\"\"\n",
    "\n",
    "    def __init__(self, node_counts, max_nodes_per_batch=20000, shuffle=True, seed=0):\n",
    "        self.node_counts = node_counts\n",
    "        self.max_nodes_per_batch = max_nodes_per_batch\n",
    "        self.shuffle = shuffle\n",
    "        self.seed = seed\n",
    "        self._epoch = 0\n",
    "\n",
    "    def __iter__(self):\n",
    "        indices = list(range(len(self.node_counts)))\n",
    "        if self.shuffle:\n",
    "            r = random.Random(self.seed + self._epoch)\n",
    "            r.shuffle(indices)\n",
    "            self._epoch += 1\n",
    "\n",
    "        batch, current = [], 0\n",
    "        for idx in indices:\n",
    "            n = self.node_counts[idx]\n",
    "            if batch and current + n > self.max_nodes_per_batch:\n",
    "                yield batch\n",
    "                batch, current = [], 0\n",
    "            batch.append(idx)\n",
    "            current += n\n",
    "        if batch:\n",
    "            yield batch\n",
    "\n",
    "    def __len__(self):\n",
    "        total = sum(self.node_counts)\n",
    "        return max(1, total // self.max_nodes_per_batch)\n",
    "\n",
    "\n",
    "def get_node_counts(variant_df, graph_dir):\n",
    "    \"\"\"Protein length (node count) for each variant, read from its cached graph - no extra computation.\"\"\"\n",
    "    cache = {}\n",
    "    counts = []\n",
    "    for uid in variant_df[\"UniProt_ID\"]:\n",
    "        if uid not in cache:\n",
    "            g = torch.load(os.path.join(graph_dir, f\"{uid}.pt\"), weights_only=False)\n",
    "            cache[uid] = g.num_residues\n",
    "        counts.append(cache[uid])\n",
    "    return counts\n",
    "\n",
    "\n",
    "def make_loaders(train_df, val_df, test_df, max_nodes_per_batch=20000):\n",
    "    \"\"\"Builds the three DataLoaders for one split - reused for the main split and each CV fold.\"\"\"\n",
    "    train_ds = MissenseVariantGraphDataset(train_df, GRAPH_DIR)\n",
    "    val_ds = MissenseVariantGraphDataset(val_df, GRAPH_DIR)\n",
    "    test_ds = MissenseVariantGraphDataset(test_df, GRAPH_DIR)\n",
    "    train_loader = DataLoader(train_ds, batch_sampler=NodeBudgetBatchSampler(\n",
    "        get_node_counts(train_df, GRAPH_DIR), max_nodes_per_batch, shuffle=True, seed=42))\n",
    "    val_loader = DataLoader(val_ds, batch_sampler=NodeBudgetBatchSampler(\n",
    "        get_node_counts(val_df, GRAPH_DIR), max_nodes_per_batch, shuffle=False))\n",
    "    test_loader = DataLoader(test_ds, batch_sampler=NodeBudgetBatchSampler(\n",
    "        get_node_counts(test_df, GRAPH_DIR), max_nodes_per_batch, shuffle=False))\n",
    "    return train_loader, val_loader, test_loader\n",
    "\n",
    "\n",
    "MAX_NODES_PER_BATCH = 20000\n",
    "train_loader, val_loader, test_loader = make_loaders(train_df, val_df, test_df, MAX_NODES_PER_BATCH)\n",
    "print(f\"Approx. batches per epoch - train: {len(train_loader)}, val: {len(val_loader)}, test: {len(test_loader)}\")"
))

# ============================================================ 5. Model architecture
cells.append(md(
    "## 5. Model Architecture\n",
    "### The GATv2 layer and PAE-weighting\n"
    "`GATv2Conv` (Brody et al., 2022) accepts an `edge_dim` argument, letting each edge's feature vector — here, "
    "`[Cα distance, PAE-derived confidence]` from Phase 3 — directly influence the attention score the model learns "
    "for that edge.\n\n"
    "### Reading out a *variant* prediction from a *protein-wide* graph\n"
    "Each graph covers the whole protein, but we only need one prediction: is *this specific substitution* "
    "pathogenic? After message passing, we read out the embedding of only the mutated residue's node and feed that "
    "single vector into a small classifier head — rather than pooling over the entire graph, which would dilute the "
    "variant's local signal across a (sometimes 1,000+ residue) protein.\n\n"
    "**Implementation note (validated empirically before use):** PyTorch Geometric batches multiple variable-sized "
    "graphs into one combined graph, offsetting node indices per graph (`batch.ptr`). We recover each graph's "
    "mutated-residue embedding via `batch.ptr[:-1] + variant_node_idx`, confirmed to produce *bit-for-bit identical* "
    "output to running each graph through the model one at a time.\n\n"
    "### Where the model classes actually live\n"
    "`GATv2Classifier`, `GCNClassifier`, and `GraphSAGEClassifier` are defined in `models.py`, not pasted into this "
    "notebook, so a saved checkpoint can be reloaded later by importing the exact same class definition used during "
    "training.\n\n"
    "### The ablation and baselines\n"
    "- **GATv2 (unweighted)**: identical code, `edge_dim=None` — isolates the effect of PAE-weighting specifically.\n"
    "- **GCN**: a non-attention convolution baseline — tests whether attention itself matters.\n"
    "- **GraphSAGE**: a second, architecturally different non-attention baseline (mean-aggregation over sampled "
    "neighbors, Hamilton et al., 2017) — broadens the comparison beyond a single non-attention architecture, so a "
    "GATv2 win (or loss) isn't an artifact of comparing against only one alternative.\n"
    "- **XGBoost (Section 7)**: no graph at all — tests whether structure matters at all."
))

# ============================================================ 6. Training setup
cells.append(md(
    "## 6. Handling Class Imbalance & Training Setup\n",
    "### Why AUPRC, not accuracy, drives model selection\n"
    "Our classes are imbalanced (~40% pathogenic / 60% benign). **AUPRC** is far more informative under imbalance "
    "than accuracy. It is therefore the primary metric for model selection and early stopping; AUROC, Precision, "
    "Recall, and F1 are reported alongside.\n\n"
    "### Loss function\n"
    "Binary cross-entropy with a `pos_weight` term from the training split's class ratio, applied identically across "
    "every model.\n\n"
    "### Early stopping, calibrated not to cut training short\n"
    "Up to 60 epochs, stopping only after 10 consecutive epochs with no validation AUPRC improvement, evaluated "
    "using the *best* validation-AUPRC checkpoint.\n\n"
    "### A practical constraint that shapes the rest of this notebook\n"
    "We measured one full training epoch at **~31 minutes** on this hardware. Every subsequent design choice in this "
    "notebook (a *constrained* hyperparameter search rather than an exhaustive grid; 3-fold rather than 5-fold "
    "cross-validation, applied only to the model the paper's claim rests on) is calibrated against this real number, "
    "not against what would be ideal with unlimited compute."
))

cells.append(code(
    "train_pos = (train_df[\"Target\"] == 1).sum()\n",
    "train_neg = (train_df[\"Target\"] == 0).sum()\n",
    "pos_weight_value = train_neg / train_pos\n",
    "print(f\"Training class balance: {train_pos:,} pathogenic / {train_neg:,} benign -> pos_weight = {pos_weight_value:.3f}\")\n",
    "POS_WEIGHT = torch.tensor([pos_weight_value], dtype=torch.float, device=DEVICE)"
))

cells.append(code(
    "def make_pos_weight(train_df):\n",
    "    pos = (train_df[\"Target\"] == 1).sum()\n",
    "    neg = (train_df[\"Target\"] == 0).sum()\n",
    "    return torch.tensor([neg / pos], dtype=torch.float, device=DEVICE)\n",
    "\n",
    "\n",
    "def run_epoch(model, loader, pos_weight, optimizer=None):\n",
    "    \"\"\"One pass over `loader`. Trains if optimizer is given, otherwise runs in eval mode.\"\"\"\n",
    "    is_train = optimizer is not None\n",
    "    model.train() if is_train else model.eval()\n",
    "\n",
    "    losses, all_logits, all_labels = [], [], []\n",
    "    with torch.set_grad_enabled(is_train):\n",
    "        for batch in loader:\n",
    "            batch = batch.to(DEVICE)\n",
    "            if is_train:\n",
    "                optimizer.zero_grad()\n",
    "            logits = model(batch)\n",
    "            loss = nn.functional.binary_cross_entropy_with_logits(logits, batch.y, pos_weight=pos_weight)\n",
    "            if is_train:\n",
    "                loss.backward()\n",
    "                optimizer.step()\n",
    "            losses.append(loss.item())\n",
    "            all_logits.extend(logits.detach().cpu().numpy().tolist())\n",
    "            all_labels.extend(batch.y.detach().cpu().numpy().tolist())\n",
    "\n",
    "    auprc = average_precision_score(all_labels, all_logits)\n",
    "    auroc = roc_auc_score(all_labels, all_logits)\n",
    "    return float(np.mean(losses)), auprc, auroc\n",
    "\n",
    "\n",
    "def train_model(model, model_name, train_loader, val_loader, pos_weight, n_epochs=60, patience=10,\n",
    "                lr=1e-3, weight_decay=1e-5, save_checkpoint=True, verbose=True, resume=True):\n",
    "    checkpoint_path = os.path.join(\"models\", f\"{model_name}_best.pt\")\n",
    "    history_path = f\"phase4_training_history_{model_name}.csv\"\n",
    "\n",
    "    # Resumability: given this notebook's total runtime (measured ~31 min/epoch), an interruption partway\n",
    "    # through is a real risk across multiple sessions. If a finished run for this exact model_name already\n",
    "    # exists, skip retraining entirely and just reload it - the same skip-if-cached pattern used for\n",
    "    # AlphaFold downloads and ESM-2 embeddings in Phases 2-3.\n",
    "    # IMPORTANT: we require BOTH the checkpoint AND the history file to exist, not just the checkpoint.\n",
    "    # The checkpoint is saved on every improving epoch (so a run interrupted mid-training - e.g. a system\n",
    "    # crash - can leave a checkpoint behind that is only a partial, possibly under-trained snapshot).\n",
    "    # history_path is only written once at the very end of a successfully completed training loop, so its\n",
    "    # presence is the real signal that this model actually finished, not just that it got partway through.\n",
    "    if resume and save_checkpoint and os.path.exists(checkpoint_path) and os.path.exists(history_path):\n",
    "        model.to(DEVICE)\n",
    "        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))\n",
    "        if verbose:\n",
    "            print(f\"[{model_name}] Found existing checkpoint - skipping training and reloading it. \"\n",
    "                  f\"Delete {checkpoint_path} first if you want to retrain from scratch.\")\n",
    "        history_df = pd.read_csv(history_path) if os.path.exists(history_path) else pd.DataFrame()\n",
    "        best_val_auprc = history_df[\"val_auprc\"].max() if len(history_df) else float(\"nan\")\n",
    "        return model, history_df, best_val_auprc\n",
    "\n",
    "    model.to(DEVICE)\n",
    "    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)\n",
    "    best_val_auprc = -1.0\n",
    "    epochs_without_improvement = 0\n",
    "    history = []\n",
    "\n",
    "    for epoch in range(1, n_epochs + 1):\n",
    "        train_loss, train_auprc, train_auroc = run_epoch(model, train_loader, pos_weight, optimizer)\n",
    "        val_loss, val_auprc, val_auroc = run_epoch(model, val_loader, pos_weight, optimizer=None)\n",
    "        history.append({\"epoch\": epoch, \"train_loss\": train_loss, \"train_auprc\": train_auprc,\n",
    "                         \"train_auroc\": train_auroc, \"val_loss\": val_loss, \"val_auprc\": val_auprc,\n",
    "                         \"val_auroc\": val_auroc})\n",
    "        if verbose:\n",
    "            print(f\"[{model_name}] epoch {epoch:2d}: train_loss={train_loss:.4f} train_AUPRC={train_auprc:.3f} \"\n",
    "                  f\"| val_loss={val_loss:.4f} val_AUPRC={val_auprc:.3f} val_AUROC={val_auroc:.3f}\")\n",
    "\n",
    "        if val_auprc > best_val_auprc:\n",
    "            best_val_auprc = val_auprc\n",
    "            epochs_without_improvement = 0\n",
    "            if save_checkpoint:\n",
    "                torch.save(model.state_dict(), checkpoint_path)\n",
    "            else:\n",
    "                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}\n",
    "        else:\n",
    "            epochs_without_improvement += 1\n",
    "            if epochs_without_improvement >= patience:\n",
    "                if verbose:\n",
    "                    print(f\"[{model_name}] Early stopping at epoch {epoch} (no improvement for {patience} epochs)\")\n",
    "                break\n",
    "\n",
    "    if save_checkpoint:\n",
    "        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))\n",
    "    else:\n",
    "        model.load_state_dict(best_state)\n",
    "    history_df = pd.DataFrame(history)\n",
    "    if save_checkpoint:\n",
    "        history_df.to_csv(history_path, index=False)\n",
    "\n",
    "    del optimizer\n",
    "    if DEVICE == \"cuda\":\n",
    "        torch.cuda.empty_cache()  # this notebook trains many models in one session - release each model's\n",
    "        # cached GPU allocations before starting the next, rather than letting them accumulate.\n",
    "    return model, history_df, best_val_auprc\n",
    "\n",
    "\n",
    "def plot_training_curves(history_df, model_name):\n",
    "    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))\n",
    "    axes[0].plot(history_df[\"epoch\"], history_df[\"train_loss\"], label=\"train loss\")\n",
    "    axes[0].plot(history_df[\"epoch\"], history_df[\"val_loss\"], label=\"val loss\")\n",
    "    axes[0].set_xlabel(\"Epoch\"); axes[0].set_ylabel(\"Loss\"); axes[0].set_title(f\"{model_name}: Loss\")\n",
    "    axes[0].legend()\n",
    "\n",
    "    axes[1].plot(history_df[\"epoch\"], history_df[\"train_auprc\"], label=\"train AUPRC\")\n",
    "    axes[1].plot(history_df[\"epoch\"], history_df[\"val_auprc\"], label=\"val AUPRC\")\n",
    "    axes[1].set_xlabel(\"Epoch\"); axes[1].set_ylabel(\"AUPRC\"); axes[1].set_title(f\"{model_name}: AUPRC\")\n",
    "    axes[1].legend()\n",
    "    plt.tight_layout()\n",
    "    safe_name = model_name.replace(\" \", \"_\").replace(\"(\", \"\").replace(\")\", \"\").replace(\",\", \"\")\n",
    "    fig.savefig(os.path.join(FIGURES_DIR, f\"training_curves_{safe_name}.png\"), dpi=200, bbox_inches=\"tight\")\n",
    "    plt.show()\n",
    "    plt.close(fig)  # this notebook produces many figures in one session - release each one rather than\n",
    "    # letting matplotlib keep every figure object alive in memory until the kernel restarts.\n",
    "\n",
    "\n",
    "def full_metrics(labels, probs, threshold=0.5):\n",
    "    preds = (np.asarray(probs) >= threshold).astype(int)\n",
    "    return {\n",
    "        \"AUPRC\": average_precision_score(labels, probs),\n",
    "        \"AUROC\": roc_auc_score(labels, probs),\n",
    "        \"Precision\": precision_score(labels, preds, zero_division=0),\n",
    "        \"Recall\": recall_score(labels, preds, zero_division=0),\n",
    "        \"F1\": f1_score(labels, preds, zero_division=0),\n",
    "    }\n",
    "\n",
    "\n",
    "def evaluate_gnn(model, loader):\n",
    "    \"\"\"Returns (labels, probabilities) for every example in loader, in evaluation mode.\"\"\"\n",
    "    model.eval()\n",
    "    all_logits, all_labels = [], []\n",
    "    with torch.no_grad():\n",
    "        for batch in loader:\n",
    "            batch = batch.to(DEVICE)\n",
    "            logits = model(batch)\n",
    "            all_logits.extend(logits.cpu().numpy().tolist())\n",
    "            all_labels.extend(batch.y.cpu().numpy().tolist())\n",
    "    probs = torch.sigmoid(torch.tensor(all_logits)).numpy()\n",
    "    return np.array(all_labels), probs"
))

# ============================================================ 7. Train models (initial pass)
cells.append(md(
    "## 7. Initial Training Pass: All Five Models, Default Hyperparameters\n",
    "These are the project's first results, with no tuning. Section 9 checks whether the differences between them are "
    "even statistically real; Sections 10-11 revisit training itself after diagnosing the overfitting visible in "
    "each GNN's training curve below."
))

cells.append(code(
    "model_pae = GATv2Classifier(in_dim=NODE_FEATURE_DIM, use_edge_attr=True)\n",
    "model_pae, history_pae, _ = train_model(model_pae, \"gatv2_pae_weighted\", train_loader, val_loader, POS_WEIGHT)\n",
    "plot_training_curves(history_pae, \"GATv2 (PAE-weighted)\")"
))

cells.append(code(
    "model_unweighted = GATv2Classifier(in_dim=NODE_FEATURE_DIM, use_edge_attr=False)\n",
    "model_unweighted, history_unweighted, _ = train_model(\n",
    "    model_unweighted, \"gatv2_unweighted\", train_loader, val_loader, POS_WEIGHT)\n",
    "plot_training_curves(history_unweighted, \"GATv2 (unweighted ablation)\")"
))

cells.append(code(
    "model_gcn = GCNClassifier(in_dim=NODE_FEATURE_DIM)\n",
    "model_gcn, history_gcn, _ = train_model(model_gcn, \"gcn_baseline\", train_loader, val_loader, POS_WEIGHT)\n",
    "plot_training_curves(history_gcn, \"GCN (baseline)\")"
))

cells.append(code(
    "model_sage = GraphSAGEClassifier(in_dim=NODE_FEATURE_DIM)\n",
    "model_sage, history_sage, _ = train_model(model_sage, \"graphsage_baseline\", train_loader, val_loader, POS_WEIGHT)\n",
    "plot_training_curves(history_sage, \"GraphSAGE (baseline)\")"
))

cells.append(md(
    "### XGBoost (Structure-Free Baseline)\n",
    "This baseline sees **no graph at all** — only the mutated residue's own ESM-2 embedding, its wild-type identity, "
    "and its substituted identity. If the GNN models substantially outperform this baseline, that is direct evidence "
    "that 3D structural context adds real predictive value beyond sequence/identity features alone."
))

cells.append(code(
    "def extract_flat_features(variant_df, graph_dir):\n",
    "    \"\"\"Per-variant feature vector: ESM-2 embedding + WT one-hot + Alt-AA one-hot at the variant's node only.\"\"\"\n",
    "    cache = {}\n",
    "    rows = []\n",
    "    for _, row in tqdm(variant_df.iterrows(), total=len(variant_df), desc=\"Extracting flat features\"):\n",
    "        uid = row[\"UniProt_ID\"]\n",
    "        if uid not in cache:\n",
    "            cache[uid] = torch.load(os.path.join(graph_dir, f\"{uid}.pt\"), weights_only=False)\n",
    "        g = cache[uid]\n",
    "        node_idx = int(row[\"Pos\"]) - 1\n",
    "        esm_emb = g.x[node_idx, :ESM2_EMBEDDING_DIM].numpy()\n",
    "        wt_onehot = g.x[node_idx, ESM2_EMBEDDING_DIM:ESM2_EMBEDDING_DIM + N_AA].numpy()\n",
    "        alt_onehot = np.zeros(N_AA, dtype=np.float32)\n",
    "        alt_onehot[AA_1TO_ONEHOT_IDX[row[\"Alt_AA_1\"]]] = 1.0\n",
    "        rows.append(np.concatenate([esm_emb, wt_onehot, alt_onehot]))\n",
    "        if len(cache) > 64:\n",
    "            cache.pop(next(iter(cache)))\n",
    "    return np.stack(rows)\n",
    "\n",
    "\n",
    "X_train = extract_flat_features(train_df, GRAPH_DIR)\n",
    "X_val = extract_flat_features(val_df, GRAPH_DIR)\n",
    "X_test = extract_flat_features(test_df, GRAPH_DIR)\n",
    "y_train, y_val, y_test = train_df[\"Target\"].values, val_df[\"Target\"].values, test_df[\"Target\"].values\n",
    "print(\"Flat feature matrix shapes:\", X_train.shape, X_val.shape, X_test.shape)"
))

cells.append(code(
    "import xgboost as xgb\n",
    "\n",
    "xgb_model = xgb.XGBClassifier(\n",
    "    n_estimators=300, max_depth=5, learning_rate=0.05,\n",
    "    scale_pos_weight=pos_weight_value, eval_metric=\"logloss\", early_stopping_rounds=20, random_state=42,\n",
    ")\n",
    "xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)\n",
    "xgb_val_probs = xgb_model.predict_proba(X_val)[:, 1]\n",
    "xgb_test_probs = xgb_model.predict_proba(X_test)[:, 1]\n",
    "print(\"XGBoost val :\", full_metrics(y_val, xgb_val_probs))\n",
    "print(\"XGBoost test:\", full_metrics(y_test, xgb_test_probs))"
))

# ============================================================ 8. Initial comparison
cells.append(md(
    "## 8. Initial Comparison: AUPRC, AUROC, Precision, Recall, F1\n",
    "Each GNN was restored from its best validation-AUPRC checkpoint before evaluation. Precision/Recall/F1 are "
    "reported at the default 0.5 probability threshold."
))

cells.append(code(
    "test_labels_pae, test_probs_pae = evaluate_gnn(model_pae, test_loader)\n",
    "test_labels_unw, test_probs_unw = evaluate_gnn(model_unweighted, test_loader)\n",
    "test_labels_gcn, test_probs_gcn = evaluate_gnn(model_gcn, test_loader)\n",
    "test_labels_sage, test_probs_sage = evaluate_gnn(model_sage, test_loader)\n",
    "# all loaders iterate test_ds in the same fixed (non-shuffled) order, so labels align across models\n",
    "assert (np.array_equal(test_labels_pae, test_labels_unw) and np.array_equal(test_labels_pae, test_labels_gcn)\n",
    "        and np.array_equal(test_labels_pae, test_labels_sage))\n",
    "\n",
    "initial_results = []\n",
    "for name, probs in [(\"GATv2 (PAE-weighted)\", test_probs_pae),\n",
    "                     (\"GATv2 (unweighted ablation)\", test_probs_unw),\n",
    "                     (\"GCN (baseline)\", test_probs_gcn),\n",
    "                     (\"GraphSAGE (baseline)\", test_probs_sage),\n",
    "                     (\"XGBoost (structure-free baseline)\", xgb_test_probs)]:\n",
    "    initial_results.append({\"model\": name, **full_metrics(test_labels_pae, probs)})\n",
    "\n",
    "initial_results_df = pd.DataFrame(initial_results).sort_values(\"AUPRC\", ascending=False)\n",
    "initial_results_df.to_csv(\"phase4_results_initial.csv\", index=False)\n",
    "print(initial_results_df.to_string(index=False))"
))

# ============================================================ 9. Statistical significance testing
cells.append(md(
    "## 9. Are These Differences Even Statistically Real?\n",
    "The models above differ by only ~0.01-0.02 AUPRC from each other. Before treating any of that as evidence about "
    "PAE-weighting or structure, we need to know whether it's a real difference or noise from which 6,771 variants "
    "happened to land in this particular test set.\n\n"
    "### Why a *paired* test, specifically\n"
    "All models were scored on the exact same test variants - their predictions are correlated. A paired test "
    "exploits that correlation directly, giving a more powerful answer than comparing independent confidence "
    "intervals.\n\n"
    "### Three complementary tests\n"
    "- **McNemar's test** compares binary correct/incorrect calls on the same instances - coarse, threshold-dependent.\n"
    "- **Paired bootstrap on the AUPRC difference** resamples the test set with replacement, computing "
    "`AUPRC(A) - AUPRC(B)` each time - uses the full ranking, not one threshold.\n"
    "- **DeLong's test** (DeLong et al., 1988) is the standard paired test for two correlated AUROC curves.\n\n"
    "We report all three because they can disagree, and disagreement is informative: rank-based tests (bootstrap, "
    "DeLong) have more statistical power than a threshold-based test (McNemar's)."
))

cells.append(code(
    "def compute_midrank(x):\n",
    "    J = np.argsort(x)\n",
    "    Z = x[J]\n",
    "    N = len(x)\n",
    "    T = np.zeros(N, dtype=float)\n",
    "    i = 0\n",
    "    while i < N:\n",
    "        j = i\n",
    "        while j < N and Z[j] == Z[i]:\n",
    "            j += 1\n",
    "        T[i:j] = 0.5 * (i + j - 1) + 1\n",
    "        i = j\n",
    "    T2 = np.empty(N, dtype=float)\n",
    "    T2[J] = T\n",
    "    return T2\n",
    "\n",
    "\n",
    "def delong_test(probs_a, probs_b, labels):\n",
    "    \"\"\"\n",
    "    Paired DeLong test for two correlated AUROCs (DeLong et al., 1988).\n",
    "    IMPORTANT: midranks for the combined array must be computed with all positives ordered before all\n",
    "    negatives - the AUC formula below only holds under that ordering (caught and fixed during validation\n",
    "    by cross-checking against sklearn's roc_auc_score, asserted below).\n",
    "    \"\"\"\n",
    "    labels = np.asarray(labels)\n",
    "    order = np.argsort(-labels)\n",
    "    m = int(np.sum(labels == 1))\n",
    "    n = int(np.sum(labels == 0))\n",
    "    preds_list = [np.asarray(probs_a), np.asarray(probs_b)]\n",
    "    k = len(preds_list)\n",
    "    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])\n",
    "    for r in range(k):\n",
    "        preds_ordered = preds_list[r][order]\n",
    "        tx[r, :] = compute_midrank(preds_ordered[:m])\n",
    "        ty[r, :] = compute_midrank(preds_ordered[m:])\n",
    "        tz[r, :] = compute_midrank(preds_ordered)\n",
    "    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n\n",
    "    v01 = (tz[:, :m] - tx[:, :]) / n\n",
    "    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m\n",
    "    delongcov = np.cov(v01) / m + np.cov(v10) / n\n",
    "\n",
    "    assert abs(aucs[0] - roc_auc_score(labels, probs_a)) < 1e-6, \"DeLong AUC mismatch vs sklearn - do not trust result\"\n",
    "    assert abs(aucs[1] - roc_auc_score(labels, probs_b)) < 1e-6, \"DeLong AUC mismatch vs sklearn - do not trust result\"\n",
    "\n",
    "    l = np.array([1, -1])\n",
    "    z = l.dot(aucs) / np.sqrt(l.dot(delongcov).dot(l))\n",
    "    p = 2 * (1 - stats.norm.cdf(abs(z)))\n",
    "    return {\"auc_a\": aucs[0], \"auc_b\": aucs[1], \"z\": z, \"p_value\": p}\n",
    "\n",
    "\n",
    "def paired_bootstrap_auprc(probs_a, probs_b, labels, n_boot=2000, seed=0):\n",
    "    rng_b = np.random.RandomState(seed)\n",
    "    labels = np.asarray(labels); probs_a = np.asarray(probs_a); probs_b = np.asarray(probs_b)\n",
    "    n_test = len(labels)\n",
    "    diffs = np.empty(n_boot)\n",
    "    for i in range(n_boot):\n",
    "        idx = rng_b.randint(0, n_test, n_test)\n",
    "        diffs[i] = (average_precision_score(labels[idx], probs_a[idx])\n",
    "                    - average_precision_score(labels[idx], probs_b[idx]))\n",
    "    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])\n",
    "    p_value = 2 * min((diffs > 0).mean(), (diffs < 0).mean())\n",
    "    return {\"mean_diff\": diffs.mean(), \"ci_low\": ci_low, \"ci_high\": ci_high, \"p_value\": p_value}\n",
    "\n",
    "\n",
    "def mcnemar_test(probs_a, probs_b, labels, threshold=0.5):\n",
    "    preds_a = (np.asarray(probs_a) >= threshold).astype(int)\n",
    "    preds_b = (np.asarray(probs_b) >= threshold).astype(int)\n",
    "    correct_a = preds_a == labels\n",
    "    correct_b = preds_b == labels\n",
    "    table = [[int(np.sum(correct_a & correct_b)), int(np.sum(correct_a & ~correct_b))],\n",
    "             [int(np.sum(~correct_a & correct_b)), int(np.sum(~correct_a & ~correct_b))]]\n",
    "    result = mcnemar(table, exact=False, correction=True)\n",
    "    return {\"table\": table, \"statistic\": result.statistic, \"p_value\": result.pvalue}\n",
    "\n",
    "\n",
    "def run_all_significance_tests(name_a, probs_a, name_b, probs_b, labels):\n",
    "    mc = mcnemar_test(probs_a, probs_b, labels)\n",
    "    bs = paired_bootstrap_auprc(probs_a, probs_b, labels)\n",
    "    dl = delong_test(probs_a, probs_b, labels)\n",
    "    print(f\"\\n=== {name_a}  vs.  {name_b} ===\")\n",
    "    print(f\"McNemar's test           : statistic={mc['statistic']:.3f}, p={mc['p_value']:.4f}\")\n",
    "    print(f\"Paired bootstrap (AUPRC) : diff={bs['mean_diff']:+.4f}, 95% CI=[{bs['ci_low']:+.4f}, {bs['ci_high']:+.4f}], p={bs['p_value']:.4f}\")\n",
    "    print(f\"DeLong's test (AUROC)    : z={dl['z']:.3f}, p={dl['p_value']:.4f}\")\n",
    "    significant = bs[\"p_value\"] < 0.05 or dl[\"p_value\"] < 0.05\n",
    "    print(f\"-> {'STATISTICALLY SIGNIFICANT' if significant else 'NOT statistically significant'} \"\n",
    "          f\"difference between {name_a} and {name_b} (alpha=0.05, using the more powerful rank-based tests)\")\n",
    "    return {\"comparison\": f\"{name_a} vs {name_b}\", \"mcnemar_p\": mc[\"p_value\"],\n",
    "            \"bootstrap_auprc_diff\": bs[\"mean_diff\"], \"bootstrap_ci_low\": bs[\"ci_low\"],\n",
    "            \"bootstrap_ci_high\": bs[\"ci_high\"], \"bootstrap_p\": bs[\"p_value\"],\n",
    "            \"delong_z\": dl[\"z\"], \"delong_p\": dl[\"p_value\"], \"significant\": significant}"
))

cells.append(code(
    "significance_results = []\n",
    "significance_results.append(run_all_significance_tests(\n",
    "    \"GATv2 (PAE-weighted)\", test_probs_pae, \"GATv2 (unweighted ablation)\", test_probs_unw, test_labels_pae))\n",
    "significance_results.append(run_all_significance_tests(\n",
    "    \"GATv2 (PAE-weighted)\", test_probs_pae, \"GraphSAGE (baseline)\", test_probs_sage, test_labels_pae))\n",
    "significance_results.append(run_all_significance_tests(\n",
    "    \"GATv2 (PAE-weighted)\", test_probs_pae, \"XGBoost (structure-free baseline)\", xgb_test_probs, test_labels_pae))\n",
    "significance_results.append(run_all_significance_tests(\n",
    "    \"GATv2 (unweighted ablation)\", test_probs_unw, \"XGBoost (structure-free baseline)\", xgb_test_probs, test_labels_pae))\n",
    "\n",
    "pd.DataFrame(significance_results).to_csv(\"phase4_significance_tests_initial.csv\", index=False)\n",
    "print(\"\\nSaved phase4_significance_tests_initial.csv\")"
))

# ============================================================ 10. Hyperparameter search
cells.append(md(
    "## 10. Hyperparameter Search for the PAE-Weighted Model\n",
    "Section 7's training curves showed validation AUPRC peaking within 1-3 epochs while training AUPRC climbed "
    "toward ~0.999 — the model memorizes faster than it generalizes at the default settings.\n\n"
    "### Why a small, short-budget search, not an exhaustive grid\n"
    "A measured **~31 minutes per full epoch** on this hardware makes a true exhaustive grid search (dozens of "
    "configurations, each trained to convergence) infeasible inside this project's timeline — that alone would take "
    "longer than is available. Instead we evaluate **6 representative configurations** spanning learning rate, "
    "dropout, hidden dimension, and weight decay, each trained for a **short, fixed budget of 5 epochs** rather than "
    "to full convergence. This is methodologically defensible specifically *because* of what Section 7 already "
    "showed: validation AUPRC peaked within the first 1-3 epochs for every GNN, so 5 epochs is enough to observe "
    "each configuration's *early* generalization behaviour — the exact thing we are searching for — without paying "
    "for a full 60-epoch run per candidate.\n\n"
    "### Why only the PAE-weighted model is searched\n"
    "Searching separately per model would multiply cost further. Instead, Section 11 applies the single winning "
    "configuration found here to *both* the PAE-weighted model and its unweighted ablation — keeping the central "
    "comparison fair (neither model gets bespoke tuning the other lacks) while only paying the search cost once."
))

cells.append(code(
    "# Saves best_config after search; reloads on restart to skip the ~15h search\n",
    "BEST_CONFIG_PATH = \"best_config.json\"\n",
    "if os.path.exists(BEST_CONFIG_PATH):\n",
    "    with open(BEST_CONFIG_PATH) as f:\n",
    "        best_config = json.load(f)\n",
    "    print(f\"Loaded best_config from {BEST_CONFIG_PATH} -- skipping hyperparameter search.\")\n",
    "    print(f\"  Winning config: {best_config['name']}  hidden_dim={int(best_config['hidden_dim'])}  \"\n",
    "          f\"dropout={best_config['dropout']:.2f}  lr={best_config['lr']}  wd={best_config['weight_decay']}\")\n",
    "    print(f\"  val AUPRC: {best_config.get('best_val_auprc', float('nan')):.4f}\")\n",
    "    print(f\"  (Delete {BEST_CONFIG_PATH} to force the search to re-run from scratch.)\")\n",
    "else:\n",
    "    search_configs = [\n",
    "        {\"name\": \"A_baseline\",      \"lr\": 1e-3, \"dropout\": 0.2, \"hidden_dim\": 128, \"weight_decay\": 1e-5},\n",
    "        {\"name\": \"B_reg_small\",     \"lr\": 3e-4, \"dropout\": 0.4, \"hidden_dim\": 64,  \"weight_decay\": 1e-4},\n",
    "        {\"name\": \"C_mid_reg\",       \"lr\": 3e-4, \"dropout\": 0.3, \"hidden_dim\": 128, \"weight_decay\": 1e-4},\n",
    "        {\"name\": \"D_low_lr_small\",  \"lr\": 1e-4, \"dropout\": 0.4, \"hidden_dim\": 64,  \"weight_decay\": 1e-4},\n",
    "        {\"name\": \"E_high_drop\",     \"lr\": 3e-4, \"dropout\": 0.4, \"hidden_dim\": 128, \"weight_decay\": 1e-5},\n",
    "        {\"name\": \"F_low_lr_light\",  \"lr\": 1e-4, \"dropout\": 0.2, \"hidden_dim\": 64,  \"weight_decay\": 1e-5},\n",
    "    ]\n",
    "\n",
    "    SEARCH_EPOCHS = 5\n",
    "    search_results = []\n",
    "    for cfg in search_configs:\n",
    "        print(f\"\\n--- Search trial {cfg['name']}: {cfg} ---\")\n",
    "        trial_model = GATv2Classifier(in_dim=NODE_FEATURE_DIM, use_edge_attr=True,\n",
    "                                       hidden_dim=cfg[\"hidden_dim\"], dropout=cfg[\"dropout\"])\n",
    "        _, _, best_val_auprc = train_model(\n",
    "            trial_model, f\"search_trial_{cfg['name']}\", train_loader, val_loader, POS_WEIGHT,\n",
    "            n_epochs=SEARCH_EPOCHS, patience=SEARCH_EPOCHS, lr=cfg[\"lr\"], weight_decay=cfg[\"weight_decay\"],\n",
    "            save_checkpoint=False, verbose=False,\n",
    "        )\n",
    "        print(f\"  -> best val AUPRC within {SEARCH_EPOCHS} epochs: {best_val_auprc:.4f}\")\n",
    "        search_results.append({**cfg, \"best_val_auprc\": best_val_auprc})\n",
    "\n",
    "    search_results_df = pd.DataFrame(search_results).sort_values(\"best_val_auprc\", ascending=False)\n",
    "    search_results_df.to_csv(\"phase4_hyperparameter_search.csv\", index=False)\n",
    "    print(\"\\n=== Search results (best first) ===\")\n",
    "    print(search_results_df.to_string(index=False))\n",
    "\n",
    "    best_config = search_results_df.iloc[0].to_dict()\n",
    "    print(f\"\\nWinning configuration: {best_config['name']}\")\n",
    "\n",
    "    # Persist to disk -- kernel restarts won't force re-running the full ~15h search\n",
    "    with open(BEST_CONFIG_PATH, \"w\") as f:\n",
    "        json.dump(best_config, f, indent=2)\n",
    "    print(f\"Saved best_config to {BEST_CONFIG_PATH}  (delete this file to re-run the search)\")\n",
))

# ============================================================ 11. Final tuned training
cells.append(md(
    "## 11. Final Training With the Winning Configuration\n",
    "Both the PAE-weighted model and its unweighted ablation are retrained from scratch using the winning "
    "configuration from Section 10, this time with the full epoch budget and early-stopping patience (not the "
    "shortened search budget) so each model is actually trained to convergence."
))

cells.append(code(
    "model_pae_final = GATv2Classifier(in_dim=NODE_FEATURE_DIM, use_edge_attr=True,\n",
    "                                    hidden_dim=int(best_config[\"hidden_dim\"]), dropout=best_config[\"dropout\"])\n",
    "model_pae_final, history_pae_final, _ = train_model(\n",
    "    model_pae_final, \"gatv2_pae_weighted_final\", train_loader, val_loader, POS_WEIGHT,\n",
    "    lr=best_config[\"lr\"], weight_decay=best_config[\"weight_decay\"])\n",
    "plot_training_curves(history_pae_final, \"GATv2 (PAE-weighted, tuned)\")"
))

cells.append(code(
    "model_unw_final = GATv2Classifier(in_dim=NODE_FEATURE_DIM, use_edge_attr=False,\n",
    "                                   hidden_dim=int(best_config[\"hidden_dim\"]), dropout=best_config[\"dropout\"])\n",
    "model_unw_final, history_unw_final, _ = train_model(\n",
    "    model_unw_final, \"gatv2_unweighted_final\", train_loader, val_loader, POS_WEIGHT,\n",
    "    lr=best_config[\"lr\"], weight_decay=best_config[\"weight_decay\"])\n",
    "plot_training_curves(history_unw_final, \"GATv2 (unweighted, tuned)\")"
))

# ============================================================ 12. Updated comparison
cells.append(md(
    "## 12. Final Comparison: All Models, Tuned Where Applicable"
))

cells.append(code(
    "test_labels_pae_final, test_probs_pae_final = evaluate_gnn(model_pae_final, test_loader)\n",
    "test_labels_unw_final, test_probs_unw_final = evaluate_gnn(model_unw_final, test_loader)\n",
    "assert np.array_equal(test_labels_pae_final, test_labels_pae)\n",
    "\n",
    "final_results = [r for r in initial_results if r[\"model\"] in\n",
    "                  (\"GCN (baseline)\", \"GraphSAGE (baseline)\", \"XGBoost (structure-free baseline)\")]\n",
    "final_results.append({\"model\": \"GATv2 (PAE-weighted, tuned)\", **full_metrics(test_labels_pae, test_probs_pae_final)})\n",
    "final_results.append({\"model\": \"GATv2 (unweighted, tuned)\", **full_metrics(test_labels_pae, test_probs_unw_final)})\n",
    "final_results.append({\"model\": \"GATv2 (PAE-weighted, original)\", **full_metrics(test_labels_pae, test_probs_pae)})\n",
    "final_results.append({\"model\": \"GATv2 (unweighted, original)\", **full_metrics(test_labels_pae, test_probs_unw)})\n",
    "\n",
    "final_results_df = pd.DataFrame(final_results).sort_values(\"AUPRC\", ascending=False)\n",
    "final_results_df.to_csv(\"phase4_results_final.csv\", index=False)\n",
    "print(final_results_df.to_string(index=False))"
))

# ============================================================ 13. Significance on final models
cells.append(md(
    "## 13. Statistical Significance Testing — Tuned Models\n",
    "The same three paired tests as Section 9, now applied to the tuned models: does PAE-weighting still favour one "
    "side once both models get the same fair tuning treatment, how do the tuned models compare to the broader "
    "baseline set, and did tuning actually change anything relative to the original run?"
))

cells.append(code(
    "significance_results_final = []\n",
    "significance_results_final.append(run_all_significance_tests(\n",
    "    \"GATv2 (PAE-weighted, tuned)\", test_probs_pae_final,\n",
    "    \"GATv2 (unweighted, tuned)\", test_probs_unw_final, test_labels_pae))\n",
    "significance_results_final.append(run_all_significance_tests(\n",
    "    \"GATv2 (PAE-weighted, tuned)\", test_probs_pae_final,\n",
    "    \"GraphSAGE (baseline)\", test_probs_sage, test_labels_pae))\n",
    "significance_results_final.append(run_all_significance_tests(\n",
    "    \"GATv2 (PAE-weighted, tuned)\", test_probs_pae_final,\n",
    "    \"XGBoost (structure-free baseline)\", xgb_test_probs, test_labels_pae))\n",
    "significance_results_final.append(run_all_significance_tests(\n",
    "    \"GATv2 (PAE-weighted, original)\", test_probs_pae,\n",
    "    \"GATv2 (PAE-weighted, tuned)\", test_probs_pae_final, test_labels_pae))\n",
    "\n",
    "pd.DataFrame(significance_results_final).to_csv(\"phase4_significance_tests_final.csv\", index=False)\n",
    "print(\"\\nSaved phase4_significance_tests_final.csv\")"
))

# ============================================================ 14. Cross-validation
cells.append(md(
    "## 14. 3-Fold Cross-Validation: How Much Does Performance Vary Across Splits?\n",
    "Every result so far comes from **one** train/val/test split. A single test-set AUPRC cannot tell you how much "
    "that number would change if a different random set of proteins had been held out — which matters directly for "
    "how confidently the paper can state a single headline number.\n\n"
    "### Why 3-fold here, not the more standard 5-fold or 10-fold\n"
    "At ~31 minutes per epoch and ~10-15 epochs to convergence per run, each fold costs roughly the same as one full "
    "training run in Section 11 (a few hours). 5-fold would cost nearly double 3-fold for a diminishing-returns "
    "improvement in the precision of the variance estimate; 3-fold is the standard, accepted compromise when full "
    "k-fold is computationally constrained, and is enough to answer the actual question here: is the test AUPRC "
    "from Section 12 a stable estimate, or did this particular split happen to be unusually easy or hard?\n\n"
    "### Why only the tuned PAE-weighted model, not every model\n"
    "This is the model the paper's central claim rests on. Running every baseline through 3-fold CV as well would "
    "triple the cost of Sections 7-13 for a question (split-to-split variance) that matters most for the one model "
    "being put forward as the contribution.\n\n"
    "### Implementation\n"
    "Each fold uses a different random seed for `make_protein_split` (so a *different* set of proteins is held out "
    "each time — verified to produce genuinely different test sets, not the same split three times), retrains a "
    "fresh model from the winning Section 10 configuration to full convergence, and evaluates on that fold's own "
    "test set. We report the mean and standard deviation across folds, not just the three individual numbers."
))

cells.append(code(
    "CV_SEEDS = [42, 123, 7]  # seed 42 matches the main split already used above, for direct comparability\n",
    "cv_results = []\n",
    "\n",
    "for seed in CV_SEEDS:\n",
    "    print(f\"\\n=== CV fold, seed={seed} ===\")\n",
    "    fold_train_df, fold_val_df, fold_test_df = make_protein_split(df, seed=seed)\n",
    "    fold_train_loader, fold_val_loader, fold_test_loader = make_loaders(fold_train_df, fold_val_df, fold_test_df)\n",
    "    fold_pos_weight = make_pos_weight(fold_train_df)\n",
    "\n",
    "    fold_model = GATv2Classifier(in_dim=NODE_FEATURE_DIM, use_edge_attr=True,\n",
    "                                  hidden_dim=int(best_config[\"hidden_dim\"]), dropout=best_config[\"dropout\"])\n",
    "    fold_model, _, _ = train_model(\n",
    "        fold_model, f\"gatv2_pae_weighted_cv_seed{seed}\", fold_train_loader, fold_val_loader, fold_pos_weight,\n",
    "        lr=best_config[\"lr\"], weight_decay=best_config[\"weight_decay\"], verbose=False)\n",
    "\n",
    "    fold_labels, fold_probs = evaluate_gnn(fold_model, fold_test_loader)\n",
    "    fold_metrics = full_metrics(fold_labels, fold_probs)\n",
    "    fold_metrics[\"seed\"] = seed\n",
    "    fold_metrics[\"n_test\"] = len(fold_labels)\n",
    "    cv_results.append(fold_metrics)\n",
    "    print(f\"  Fold result: {fold_metrics}\")\n",
    "\n",
    "cv_results_df = pd.DataFrame(cv_results)\n",
    "summary_row = {\n",
    "    \"seed\": \"mean ± std\", \"n_test\": \"-\",\n",
    "    **{metric: f\"{cv_results_df[metric].mean():.4f} ± {cv_results_df[metric].std():.4f}\"\n",
    "       for metric in [\"AUPRC\", \"AUROC\", \"Precision\", \"Recall\", \"F1\"]}\n",
    "}\n",
    "print(\"\\n=== 3-Fold CV Summary (tuned PAE-weighted model) ===\")\n",
    "print(pd.concat([cv_results_df, pd.DataFrame([summary_row])], ignore_index=True).to_string(index=False))\n",
    "cv_results_df.to_csv(\"phase4_cv_results.csv\", index=False)\n",
    "print(\"\\nSaved phase4_cv_results.csv\")"
))

# ============================================================ 15. Interpretability
cells.append(md(
    "## 15. Does the Model Understand *Why*? Error Analysis & Attention Inspection\n",
    "A high AUPRC says the model ranks pathogenic variants above benign ones well — it does not by itself say *what* "
    "the model is using to make that judgement. This section looks under the hood of the tuned PAE-weighted model "
    "from Section 11 (the main-split version, not a CV fold, so it lines up with the error-analysis variant table "
    "used elsewhere in this notebook).\n\n"
    "### 15a. Confusion matrix on the test set"
))

cells.append(code(
    "best_pae_probs = test_probs_pae_final\n",
    "best_pae_model = model_pae_final\n",
    "\n",
    "preds = (best_pae_probs >= 0.5).astype(int)\n",
    "cm = confusion_matrix(test_labels_pae, preds)\n",
    "print(\"Confusion matrix (rows=actual, cols=predicted), [[TN, FP], [FN, TP]]:\")\n",
    "print(cm)\n",
    "tn, fp, fn, tp = cm.ravel()\n",
    "print(f\"\\nPrecision={precision_score(test_labels_pae, preds):.3f}, \"\n",
    "      f\"Recall={recall_score(test_labels_pae, preds):.3f}, F1={f1_score(test_labels_pae, preds):.3f}\")"
))

cells.append(md(
    "### 15b. The model's most confident mistakes"
))

cells.append(code(
    "all_uniprot, all_varid = [], []\n",
    "for batch in test_loader:\n",
    "    all_uniprot.extend(batch.uniprot_id)\n",
    "    all_varid.extend(batch.variation_id)\n",
    "\n",
    "error_df = pd.DataFrame({\n",
    "    \"VariationID\": all_varid, \"UniProt_ID\": all_uniprot,\n",
    "    \"true_label\": test_labels_pae, \"predicted_prob\": best_pae_probs,\n",
    "})\n",
    "error_df[\"abs_error\"] = (error_df[\"true_label\"] - error_df[\"predicted_prob\"]).abs()\n",
    "\n",
    "print(\"Top 10 most confident mistakes (model very sure, but wrong):\")\n",
    "print(error_df.sort_values(\"abs_error\", ascending=False).head(10).to_string(index=False))\n",
    "\n",
    "print(\"\\nTest-set error by protein (proteins with >= 5 test variants, worst-performing first):\")\n",
    "per_protein = error_df.groupby(\"UniProt_ID\").apply(\n",
    "    lambda g: pd.Series({\"n\": len(g), \"mean_abs_error\": g[\"abs_error\"].mean()}), include_groups=False\n",
    ")\n",
    "print(per_protein[per_protein[\"n\"] >= 5].sort_values(\"mean_abs_error\", ascending=False).head(10))"
))

cells.append(md(
    "### 15c. What is the attention mechanism actually focusing on?\n",
    "For one real test-set variant, we extract GATv2's learned attention weights on the mutated residue's incoming "
    "edges and rank its structural neighbors by how much attention the model paid them."
))

cells.append(code(
    "example_row = test_df.iloc[0]\n",
    "example_ds = MissenseVariantGraphDataset(test_df.iloc[[0]], GRAPH_DIR)\n",
    "example_data = example_ds[0].to(DEVICE)\n",
    "example_batch = Batch.from_data_list([example_data])\n",
    "\n",
    "best_pae_model.eval()\n",
    "with torch.no_grad():\n",
    "    logit, (edge_index_1, alpha_1, edge_index_2, alpha_2) = best_pae_model(example_batch, return_attention=True)\n",
    "\n",
    "variant_node = example_data.variant_node_idx.item()\n",
    "incoming_mask = edge_index_1[1] == variant_node\n",
    "neighbor_nodes = edge_index_1[0][incoming_mask].cpu().numpy()\n",
    "neighbor_attention = alpha_1[incoming_mask].mean(dim=1).cpu().numpy()\n",
    "\n",
    "attn_df = pd.DataFrame({\"neighbor_residue\": neighbor_nodes + 1, \"attention_weight\": neighbor_attention})\n",
    "attn_df = attn_df.sort_values(\"attention_weight\", ascending=False)\n",
    "\n",
    "print(f\"Variant: {example_row['UniProt_ID']} {example_row['Ref_AA']}{example_row['Pos']}{example_row['Alt_AA']} \"\n",
    "      f\"(true label: {'Pathogenic' if example_row['Target']==1 else 'Benign'}, \"\n",
    "      f\"predicted probability: {torch.sigmoid(logit).item():.3f})\")\n",
    "print(f\"\\nLayer-1 attention from the mutated residue's {len(attn_df)} structural neighbors:\")\n",
    "print(attn_df.to_string(index=False))"
))

# ============================================================ 16. Future work + Methods summary
cells.append(md(
    "## 16. Not Yet Included — Flagged as Future Work\n",
    "A direct comparison against AlphaMissense's published scores on the same test variants is **not** part of this "
    "notebook. That requires sourcing AlphaMissense's precomputed score table and joining it to `test_df` by variant "
    "- a clean, separate task (Phase 5) once this model's own results are finalized.\n\n"
    "Also out of scope given the project timeline, and explicitly noted as such rather than silently skipped: an "
    "exhaustive hyperparameter grid search (Section 10 used a constrained 6-configuration search instead), 5-fold or "
    "10-fold cross-validation (Section 14 used 3-fold), and cross-validating every baseline architecture rather than "
    "only the proposed model."
))

cells.append(md(
    "## 17. Methods Summary (Copy-Paste for the Paper)\n",
    "> Proteins were split into training, validation, and test sets (70/15/15) at the protein level (by UniProt "
    "accession) to prevent data leakage between variants sharing the same structural graph; disjointness was verified "
    "programmatically. Batches were constructed using a node-budget sampler bounding total residue count per batch "
    "rather than a fixed number of graphs, keeping peak GPU memory bounded given the wide range of protein sizes (up "
    "to 2,699 residues). The proposed model was a two-layer GATv2 (Brody et al., 2022) in which edge attention was "
    "conditioned on AlphaFold's Cα distance and Predicted Aligned Error (PAE) between residue pairs; the mutated "
    "residue's post-message-passing embedding was extracted and passed to an MLP classifier rather than pooling over "
    "the full graph. Class imbalance was addressed via a positive-class weight in the loss. Four baselines isolated "
    "each architectural choice: an identical GATv2 with edge features withheld (testing PAE-weighting specifically), "
    "a GCN and a GraphSAGE model (Hamilton et al., 2017) testing the contribution of attention against two "
    "architecturally distinct non-attention alternatives, and an XGBoost classifier on the variant residue's ESM-2 "
    "embedding and amino-acid identity alone with no structural graph (testing structure itself). An initial "
    "training pass showed all models within ~0.01-0.02 AUPRC of one another, with the GNNs overfitting within 1-3 "
    "epochs. This was addressed with a constrained hyperparameter search (6 configurations spanning learning rate, "
    "dropout, hidden dimensionality, and weight decay, each evaluated over a short 5-epoch budget given the observed "
    "early overfitting) for the PAE-weighted model, with the winning configuration then applied identically to its "
    "unweighted ablation and both retrained to full convergence with early stopping. Pairwise model comparisons were "
    "evaluated for statistical significance using three paired tests on the held-out test set: McNemar's test "
    "(McNemar, 1947), a paired bootstrap test on the AUPRC difference (2,000 resamples), and DeLong's test (DeLong "
    "et al., 1988) for correlated AUROC curves. Finally, the tuned PAE-weighted model's robustness to the choice of "
    "train/test split was assessed via 3-fold cross-validation (three independent protein-level splits, each with "
    "the model retrained to convergence), reporting mean ± standard deviation across folds rather than a single "
    "split's point estimate."
))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.9.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out_path = r"c:\Users\Megha Kanojia\Documents\DevProjects\agy2-projects\missense-variants\phase4_model_training.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook written to {out_path}")
