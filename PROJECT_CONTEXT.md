# Project Context — MSc Practicum: PAE-Weighted GNN for Missense Variant Pathogenicity

## Student & Deadline
- **Student**: Megha Kanojia, MSc Data Analytics, Dublin City University
- **Deadline**: July 20, 2026 (research paper submission)
- **Supervisor**: DCU supervisor (name not recorded)

---

## Project Goal
Train a Graph Attention Network (GATv2) that uses AlphaFold's **Predicted Aligned Error (PAE)**
as edge weights to predict whether a missense variant is pathogenic or benign.
The central claim being tested: *does PAE-weighting improve over an unweighted GNN that only
uses 3D contact structure?*

---

## Dataset
- **Source**: ClinVar missense variants joined to AlphaFold protein structures
- **File**: `clinvar_graph_ready.csv` (49,303 variants, 6,597 proteins)
- **Target**: 1 = Pathogenic, 0 = Benign (~41% pathogenic)
- **Split**: Protein-level 70/15/15 train/val/test (seed=42) — no protein appears in more than one split
- **Test set**: 6,771 variants, 991 proteins

### Node features (1321-dim per residue)
| Component | Dims | Source |
|---|---|---|
| ESM-2 embedding | 1280 | Facebook ESM-2 protein language model (Phase 2) |
| Wild-type AA one-hot | 20 | ClinVar record |
| Variant flag | 1 | 1 for mutated residue, 0 for all others |
| Alternate AA one-hot | 20 | ClinVar record |

### Edge features (2-dim per contact)
| Component | Description |
|---|---|
| Cα–Cα distance | Physical distance between residues |
| PAE weight | `1/(1+PAE)` — AlphaFold confidence; practical ceiling ~0.5 (p50/p75/p95/p99 all = 0.5) |

**Key diagnostic finding**: PAE weights are heavily saturated at 0.5 (AlphaFold rarely reports
PAE < 1Å for confident pairs). This explains why PAE-weighted GATv2 did NOT outperform the
unweighted ablation.

---

## Phases Completed

### Phase 1 — Data Extraction (`phase1_data_extraction.ipynb`)
- Parsed ClinVar XML, filtered to missense variants with reviewed clinical significance
- Mapped gene symbols → UniProt IDs
- Output: `clinvar_graph_ready.csv`

### Phase 2 — Asset Download (`phase2_asset_download.ipynb`)
- Downloaded AlphaFold structures (PDB files) for all 6,597 proteins
- Computed ESM-2 (650M) embeddings for every residue in every protein
- Cached in `protein_graphs/` directory

### Phase 3 — Graph Construction (`phase3_graph_construction.ipynb`)
- Built PyTorch Geometric `Data` objects per protein
- Nodes = residues, edges = Cα–Cα contacts within threshold
- Edge features = [distance, PAE weight]
- Output: `protein_graphs/{UniProt_ID}.pt` for all 6,597 proteins

### Phase 4 — Model Training (`phase4_model_training.ipynb`)

#### Models trained
| Model | Checkpoint | Notes |
|---|---|---|
| GATv2 (PAE-weighted, original) | `gatv2_pae_weighted_best.pt` (1745 KB) | Default hyperparams |
| GATv2 (unweighted ablation, original) | `gatv2_unweighted_best.pt` (1737 KB) | No edge features |
| GCN (baseline) | `gcn_baseline_best.pt` (826 KB) | |
| GraphSAGE (baseline) | `graphsage_baseline_best.pt` (955 KB) | |
| GATv2 (PAE-weighted, tuned) | `gatv2_pae_weighted_final_best.pt` (621 KB) | B_reg_small config |
| GATv2 (unweighted, tuned) | `gatv2_unweighted_final_best.pt` (616 KB) | B_reg_small config |
| CV seed=42 | `gatv2_pae_weighted_cv_seed42_best.pt` (621 KB) | |
| CV seed=123 | `gatv2_pae_weighted_cv_seed123_best.pt` (621 KB) | |
| CV seed=7 | `gatv2_pae_weighted_cv_seed7_best.pt` (621 KB) | |

XGBoost baseline was also trained on flat ESM-2 features (no graph) — no checkpoint saved,
results in `phase4_results_initial.csv`.

#### Winning hyperparameter config (`best_config.json`)
```json
{
  "name": "B_reg_small",
  "lr": 0.0003,
  "dropout": 0.4,
  "hidden_dim": 64,
  "weight_decay": 0.0001,
  "best_val_auprc": 0.9111
}
```
Note: `best_config.json` is the authoritative source. Section 10 of the notebook loads from
this file on restart — delete the file to force the search to re-run (~15 hours).

#### Final comparison results (`phase4_results_final.csv`)
| Model | AUPRC | AUROC |
|---|---|---|
| GraphSAGE (baseline) | **0.9148** | 0.9393 |
| XGBoost (structure-free) | 0.9125 | 0.9354 |
| GATv2 (PAE-weighted, original) | 0.9112 | 0.9373 |
| GATv2 (unweighted, original) | 0.9112 | 0.9358 |
| GATv2 (unweighted, tuned) | 0.9097 | 0.9355 |
| GATv2 (PAE-weighted, tuned) | 0.9069 | 0.9339 |
| GCN (baseline) | 0.8938 | 0.9265 |

**Key finding**: PAE-weighting does NOT help — tuned PAE model has lower AUPRC than the tuned
unweighted model (bootstrap diff=-0.0028), and GraphSAGE is the best performing model overall.

**CORRECTION (2026-07-05, verified against `phase4_significance_tests_final.csv` on disk)**:
this file previously claimed the PAE-weighted-tuned vs. unweighted-tuned gap was
"SIGNIFICANT (p=0.026)". The significance-test CSV currently on disk — most recently
regenerated 2026-07-05 10:34, i.e. after that claim was written — shows this specific
comparison is **NOT significant** at alpha=0.05 (bootstrap p=0.073, DeLong p=0.067,
McNemar p=0.348). The AUPRC values themselves (0.9069 tuned-PAE vs 0.9097 tuned-unweighted)
match this document's table, so this isn't a data-loading error — a rerun of training and/or
the bootstrap resampling between when this doc's numbers were written and now shifted the
p-value across the 0.05 threshold. Given the effect size is tiny (~0.003 AUPRC) relative to
the ~0.002 CV std observed elsewhere, treat "does PAE-weighting hurt" as **directionally
consistent but not robustly significant across reruns** — this is the more defensible framing
for the paper's Discussion/Limitations section. Do not cite "p=0.026, significant" anywhere
in the write-up; use the current CSV's numbers instead.

#### Significance tests (`phase4_significance_tests_final.csv`, current as of 2026-07-05)
- PAE-weighted tuned vs unweighted tuned: **NOT significant** (bootstrap p=0.073, DeLong
  p=0.067, McNemar p=0.348; PAE lower by 0.0028 AUPRC)
- PAE-weighted tuned vs GraphSAGE: **SIGNIFICANT** (p=0.000, PAE worse by 0.0079)
- PAE-weighted tuned vs XGBoost: mixed — bootstrap/McNemar significant, DeLong not (p=0.358)
- PAE-weighted original vs PAE-weighted tuned: **SIGNIFICANT** (p=0.0098, original better)

#### 3-Fold CV results (`phase4_cv_results.csv`) — tuned PAE-weighted model
| Seed | AUPRC | n_test |
|---|---|---|
| 42 | 0.9077 | 6,771 |
| 123 | 0.9102 | 8,159 |
| 7 | 0.9054 | 6,924 |
| **mean ± std** | **0.9078 ± 0.0024** | — |

Tight std = performance is stable across different protein splits.

### Phase 5 — AlphaMissense Comparison (`phase5_alphamissense_comparison.ipynb`)

#### Setup
- AlphaMissense scores downloaded from Zenodo (DOI 10.5281/zenodo.8208688),
  file: `AlphaMissense_aa_substitutions.tsv.gz`
- Filtered to test proteins, cached as `alphamissense_test_scores.csv` (382 MB)
- Joined to test set by `(UniProt_ID, {Ref1letter}{Pos}{Alt1letter})`
- GNN predictions cached as `phase5_gnn_predictions.csv`

#### Results on matched subset (`phase5_results.csv`, updated 2026-07-05 with Accuracy column)
| Model | AUPRC | AUROC | Accuracy | Threshold |
|---|---|---|---|---|
| **AlphaMissense** | **0.9286** | **0.9478** | 0.8816 | 0.564 |
| GraphSAGE (baseline) | 0.9159 | 0.9405 | 0.8654 | 0.5 |
| GATv2 (PAE-weighted, tuned) | 0.9073 | 0.9348 | 0.8632 | 0.5 |

#### Significance tests (`phase5_significance_tests.csv`)
- GATv2 vs AlphaMissense: **SIGNIFICANT** (p=0.000, diff=-0.0214, GATv2 worse)
- GraphSAGE vs AlphaMissense: **SIGNIFICANT** (p=0.001, diff=-0.0128, GraphSAGE worse)
- GATv2 vs GraphSAGE: **SIGNIFICANT** (p=0.000, diff=-0.0086, GATv2 worse)

**Key finding**: AlphaMissense significantly outperforms both GNN models. This is expected —
AlphaMissense uses evolutionary conservation + AlphaFold structure at much larger scale.
The GNN trained only on ClinVar labels achieves comparable but not matching performance.

---

## Key Source Files

| File | Purpose |
|---|---|
| `clinvar_graph_ready.csv` | Full dataset (49,303 variants) |
| `protein_graphs/{UniProt_ID}.pt` | Per-protein graphs (6,597 files) |
| `models/*.pt` | All trained model checkpoints |
| `best_config.json` | Winning hyperparameter config (B_reg_small) |
| `missense_dataset.py` | PyTorch Dataset class, NODE_FEATURE_DIM=1321 |
| `models.py` | GATv2Classifier, GCNClassifier, GraphSAGEClassifier |
| `phase4_model_training.ipynb` | Main training notebook (Sections 1–17) |
| `phase5_alphamissense_comparison.ipynb` | AlphaMissense comparison notebook (Sections 1–10; 9–10 added 2026-07-05) |
| `gnn_dataset_overview.ipynb` | Supervisor presentation notebook |
| `generate_nb_phase4.py` | Generator script for phase4 notebook |
| `generate_nb_phase5.py` | Generator script for phase5 notebook |
| `alphamissense_test_scores.csv` | Cached AlphaMissense scores for test proteins |
| `phase5_gnn_predictions.csv` | Cached GNN predictions on test set |
| `pae_value_quantification.csv` | **New (2026-07-05)** — proposal-required %-change-in-AUPRC table, PAE-weighted vs. unweighted, original and tuned stages |

---

## Architecture Details

### GATv2Classifier (from `models.py`)
- 2-layer GATv2Conv with `edge_dim=2` (PAE-weighted) or `edge_dim=None` (unweighted)
- 4 attention heads per layer
- Output: single logit per variant from the **mutated residue's node** (NOT global pooling)
- `hidden_dim=64` for tuned models, `hidden_dim=128` for original models

### NodeBudgetBatchSampler
- Caps total residues per batch at 20,000 (not graphs) to prevent GPU OOM
- RTX 4050 Laptop GPU (6.4 GB VRAM)
- ~1,741 batches per epoch, ~31 minutes per epoch

### Training setup
- Loss: Binary cross-entropy with `pos_weight=1.470` (neg/pos ratio)
- Optimizer: Adam
- Early stopping: patience=10, up to 60 epochs
- Primary metric: AUPRC (class imbalance makes accuracy/AUROC less informative)

---

## Narrative for the Paper

### What the results say
1. **Does structure help?** — Marginally. GNNs and XGBoost are within 0.002 AUPRC of each
   other, and differences are not consistently significant in initial tests. Structure adds
   negligible value over flat ESM-2 features alone on this dataset.

2. **Does PAE-weighting help?** — No. The PAE-weighted model has lower AUPRC than the
   unweighted ablation after tuning (-0.0028, i.e. -0.31%; see `pae_value_quantification.csv`),
   though this specific gap is **not statistically significant** on the current run (bootstrap
   p=0.073, DeLong p=0.067 — see correction note in Phase 4 section above). The PAE signal is
   saturated: p50/p75/p95 of edge weights all = 0.5, leaving very little variation for the
   attention mechanism to exploit — this is a plausible mechanism regardless of whether the
   AUPRC gap itself clears significance.

3. **How does it compare to AlphaMissense?** — AlphaMissense is significantly better
   (AUPRC 0.9286 vs 0.9073, p<0.001). A ClinVar-trained GNN cannot match a model trained
   at scale with evolutionary conservation signals.

4. **Is the result stable?** — Yes. CV std=0.0024 across 3 folds.

### Honest interpretation
The project's central hypothesis (PAE-weighting helps) was **not supported**. This is a valid
scientific finding — the PAE saturation diagnostic (Section 2) provides a clear mechanistic
explanation. The paper should frame this as a rigorous negative result rather than a failure.

---

## Pending / Optional Work
- **CORRECTION (2026-07-05)**: Section 15 (attention visualization) is already complete with
  real outputs (confusion matrix, top mistakes, attention weights on variant P06132
  Gly281Val) — the previous note here claiming it still needed to be run was stale.
- **Phase 5 figure** already saved to `figures/phase5_comparison.png`.
- Paper methods draft paragraphs exist in Section 17 of `phase4_model_training.ipynb`, Section
  11 of `phase3_graph_construction.ipynb`, and Section 8 of `phase5_alphamissense_comparison.ipynb`
  — but **no assembled paper document exists yet** (no .docx/.tex/.md draft with
  Introduction/Results/Discussion found in the project directory as of 2026-07-05). This is
  the critical-path item before the July 20, 2026 submission, not further experiments.
- **Added 2026-07-05**: two proposal-required deliverables that were missing were added as new
  cells (Sections 9–10) in `phase5_alphamissense_comparison.ipynb`:
  - **Accuracy** (secondary metric named in the proposal, never computed before) — for the
    3 models compared in Phase 5 only (AlphaMissense 0.8816, GraphSAGE 0.8654, GATv2
    PAE-weighted tuned 0.8632). Written into `phase5_results.csv`. Accuracy for the other 4
    Phase 4 models (GCN, XGBoost, unweighted variants) could not be added retroactively —
    their raw test-set predictions were never cached, only aggregate metrics.
  - **PAE Value Quantification** (the proposal's own named deliverable: %-change in AUPRC,
    PAE-weighted vs. unweighted) — written to `pae_value_quantification.csv`. Tuned pair:
    -0.31% (not significant, see correction above). Original/untuned pair: +0.004% (not
    significant, p=0.94/0.27).

---

## Important Technical Notes
- `best_config.json` is the restart-safe store of the winning hyperparameters. Never delete
  it unless you want to re-run the 15h search.
- All Phase 4 checkpoints are saved in `models/`. Original models use `hidden_dim=128`,
  tuned models use `hidden_dim=64` (B_reg_small). Do NOT mix checkpoints between them.
- `train_model()` requires BOTH checkpoint `.pt` AND history `.csv` to exist to skip
  retraining — a checkpoint alone is not enough (it could be from a crashed partial run).
- The `Ref_AA` column in the dataset uses 3-letter codes (e.g. "Gly"). `Alt_AA_1` is
  already 1-letter. AlphaMissense join key = `{Ref_AA_1letter}{Pos}{Alt_AA_1}`.
