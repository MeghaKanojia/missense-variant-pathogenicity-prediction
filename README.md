# Identifying Pathogenic Significance Using AlphaFold2 3D Structures

MSc Data Analytics practicum, Dublin City University.
Authors: Megha Kanojia, Aadesh Sunil Kshetre. Supervisor: Andrew McCarren.

Can a graph neural network that weights protein contact edges by AlphaFold's **Predicted Aligned Error (PAE)** predict missense variant pathogenicity better than an unweighted GNN, standard baselines, or AlphaMissense?

**Short answer: no.** PAE-weighting gave no measurable benefit across three different PAE encodings. The full write-up is in [`paper/Final Research Paper.pdf`](paper/Final%20Research%20Paper.pdf).

## Approach

- **Data:** 49,820 ClinVar missense variants (2-star+ review status, germline), 6,597 proteins after structure matching.
- **Graphs:** one graph per protein from its AlphaFold structure. Nodes are residues with ESM-2 (`esm2_t33_650M_UR50D`) embeddings plus wild-type/mutant one-hot and a variant-site flag (1321 dims). Edges connect residues with Cα-Cα distance < 8 Å, with edge features `[distance, 1/(1+PAE)]`.
- **Model:** 2-layer GATv2 (4 heads), prediction read from the mutated residue's embedding.
- **Baselines:** unweighted GATv2 (ablation), GCN, GraphSAGE, XGBoost (structure-free), AlphaMissense.
- **Evaluation:** protein-level 70/15/15 split (no protein in more than one split), AUPRC as primary metric, McNemar, bootstrap and DeLong significance tests, 3-fold CV.

## Results (test set)

| Model | AUPRC | AUROC | F1 |
|---|---|---|---|
| AlphaMissense | 0.9286 | 0.9478 | 0.8407 |
| GraphSAGE (baseline) | 0.9148 | 0.9393 | 0.8405 |
| XGBoost (structure-free) | 0.9125 | 0.9354 | 0.8305 |
| GATv2, PAE-weighted (untuned) | 0.9112 | 0.9373 | 0.8407 |
| GATv2, unweighted (tuned) | 0.9097 | 0.9355 | 0.8354 |
| GATv2, PAE-weighted (tuned) | 0.9069 | 0.9339 | 0.8313 |
| GCN (baseline) | 0.8938 | 0.9265 | 0.8104 |

Tuned PAE-weighted vs. tuned unweighted: -0.31% AUPRC, not significant (bootstrap p=0.073, DeLong p=0.067). Two follow-up experiments (per-protein min-max PAE normalisation, and directional PAE features keeping `PAE[i][j]` and `PAE[j][i]` separate) gave the same result. A likely cause is saturation: most confident PAE values sit at AlphaFold's ~1 Å floor, so `1/(1+PAE)` is tied at 0.5 for over half of all edges.

## Repository layout

| Path | Contents |
|---|---|
| `phase1_data_extraction.ipynb` | ClinVar filtering → `clinvar_missense_filtered.csv` |
| `phase2_asset_download.ipynb` | AlphaFold PDB + PAE download → `alphafold_data/`, `clinvar_with_structures.csv` |
| `phase3_graph_construction.ipynb` | ESM-2 embeddings + PyG graphs → `esm2_embeddings/`, `protein_graphs/`, `clinvar_graph_ready.csv` |
| `phase4_model_training.ipynb` | Training, tuning, CV, significance tests, error analysis |
| `phase5_alphamissense_comparison.ipynb` | AlphaMissense head-to-head, accuracy, PAE value quantification |
| `experiment_pae_*.ipynb`, `pae_*_experiment/` | Supplementary PAE encoding experiments (results, figures, models) |
| `models.py`, `missense_dataset.py` | Shared model and dataset code |
| `generate_nb_*.py` | Scripts that generated the phase notebooks |
| `models/` | Trained checkpoints for all GNN models |
| `figures/`, `paper/` | Figures, LaTeX source, final paper PDF, slides |
| `*.csv` | Filtered datasets, manifests, results and significance tests |

## Rebuilding the data

About 113 GB of intermediate data is not in this repo. Everything left out is public or derived from public data, and can be rebuilt by running the notebooks in order:

1. **Raw downloads** (place in the repo root):
   - ClinVar `variant_summary.txt.gz` from <https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/>. This project used the **June 2026** release (archived monthly releases are under `archive/`). Not needed if you start from the committed `clinvar_missense_filtered.csv`, which freezes the dataset.
   - `AlphaMissense_aa_substitutions.tsv.gz` from the AlphaMissense Zenodo record (<https://zenodo.org/records/8208688>). Needed for Phase 5.
2. **Phase 2** re-downloads AlphaFold structures and PAE files (~16 GB) via the AlphaFold DB API, driven by `asset_download_manifest.csv`. `alphafold_data/metadata/` is committed and records the exact model versions used; AlphaFold DB versions may have moved on since.
3. **Phase 3** recomputes ESM-2 embeddings (~23 GB, GPU recommended) and protein graphs (~25 GB).
4. **Experiment notebooks** rebuild their own graph variants (~25 GB each) from the Phase 3 outputs.

Phase 4/5 results can be reproduced from the committed checkpoints in `models/` without retraining, once the graphs exist.

## Environment

Python, PyTorch 2.x (CUDA 12.6), PyTorch Geometric, transformers (ESM-2), xgboost, scikit-learn, scipy, statsmodels, pandas, numpy, matplotlib, networkx, requests, tqdm.
Trained on an NVIDIA RTX 4050 Laptop GPU (6 GB VRAM).
