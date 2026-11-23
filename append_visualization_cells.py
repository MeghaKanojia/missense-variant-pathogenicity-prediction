import json

nb_path = r"c:\Users\Megha Kanojia\Documents\DevProjects\agy2-projects\missense-variants\phase2_asset_download.ipynb"

with open(nb_path, "r") as f:
    nb = json.load(f)

def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": list(lines)}

def code(*lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": list(lines)}

new_cells = []

new_cells.append(md(
    "## 9. Visualize an Example Structure With Variants Highlighted\n",
    "As a sanity check on everything above — and a way to *see* what the pipeline produced — we render one protein's "
    "AlphaFold structure and mark where our ClinVar variants fall on it.\n\n"
    "**Confidence coloring (pLDDT):** AlphaFold stores its per-residue confidence score (pLDDT, 0–100) in the PDB "
    "file's B-factor column — this is the standard convention used by the AlphaFold DB itself (Jumper et al., 2021, "
    "[doi:10.1038/s41586-021-03819-2](https://doi.org/10.1038/s41586-021-03819-2); Tunyasuvunakool et al., 2021, "
    "[doi:10.1038/s41586-021-03828-1](https://doi.org/10.1038/s41586-021-03828-1)). We reuse AlphaFold's own four-bin "
    "color scheme:\n"
    "- **Blue** — pLDDT > 90 (very high confidence)\n"
    "- **Cyan** — pLDDT 70–90 (confident)\n"
    "- **Yellow** — pLDDT 50–70 (low confidence)\n"
    "- **Orange** — pLDDT < 50 (very low confidence — often a flexible/disordered region, not a folding error)\n\n"
    "**Variant markers:** pathogenic ClinVar variants are shown as **red** spheres, benign as **green** spheres, on "
    "top of the cartoon. This view is exactly the kind of structural context (confident, well-folded core vs. "
    "low-confidence surface regions) that the PAE-weighted GNN is intended to exploit in Phase 3."
))

new_cells.append(code(
    "!pip install py3Dmol -q"
))

new_cells.append(code(
    "import py3Dmol\n",
    "from Bio.PDB import PDBParser\n",
    "import warnings\n",
    "from Bio import BiopythonWarning\n",
    "warnings.simplefilter(\"ignore\", BiopythonWarning)"
))

new_cells.append(md(
    "### 9a. Pick an Example Protein\n",
    "We pick a protein with a good number of *both* pathogenic and benign variants, so the contrast is visible in one "
    "image. `BRCA1` (`P38398`) — already used as a worked example earlier in this notebook — fits well: 318 variants "
    "in our Phase 2 output, both classes present."
))

new_cells.append(code(
    "struct_df = pd.read_csv(\"clinvar_with_structures.csv\")\n",
    "\n",
    "counts = struct_df.groupby(\"UniProt_ID\")[\"Target\"].agg([\"count\", \"nunique\"])\n",
    "candidates = counts[(counts[\"nunique\"] == 2) & (counts[\"count\"] >= 5)].sort_values(\"count\", ascending=False)\n",
    "print(\"Top candidate proteins (most variants, both classes present):\")\n",
    "print(candidates.head(10))\n",
    "\n",
    "EXAMPLE_UNIPROT_ID = \"P38398\"  # BRCA1\n",
    "example_variants = struct_df[struct_df[\"UniProt_ID\"] == EXAMPLE_UNIPROT_ID].copy()\n",
    "print(f\"\\nUsing {EXAMPLE_UNIPROT_ID} — {len(example_variants):,} variants \"\n",
    "      f\"({(example_variants['Target']==1).sum()} pathogenic, {(example_variants['Target']==0).sum()} benign)\")"
))

new_cells.append(md(
    "### 9b. Extract Per-Residue Confidence (pLDDT) From the PDB File"
))

new_cells.append(code(
    "pdb_path = os.path.join(PDB_DIR, f\"{EXAMPLE_UNIPROT_ID}.pdb\")\n",
    "with open(pdb_path, \"r\") as f:\n",
    "    pdb_str = f.read()\n",
    "\n",
    "parser = PDBParser(QUIET=True)\n",
    "structure = parser.get_structure(EXAMPLE_UNIPROT_ID, pdb_path)\n",
    "\n",
    "def plddt_to_color(plddt):\n",
    "    if plddt > 90: return \"#0053D6\"   # blue — very high\n",
    "    if plddt > 70: return \"#65CBF3\"   # cyan — confident\n",
    "    if plddt > 50: return \"#FFDB13\"   # yellow — low\n",
    "    return \"#FF7D45\"                  # orange — very low\n",
    "\n",
    "residue_colors = {}\n",
    "for model in structure:\n",
    "    for chain in model:\n",
    "        for residue in chain:\n",
    "            ca_atoms = [a for a in residue if a.get_id() == \"CA\"]\n",
    "            if ca_atoms:\n",
    "                plddt = ca_atoms[0].get_bfactor()\n",
    "                residue_colors[residue.get_id()[1]] = plddt_to_color(plddt)\n",
    "    break  # AlphaFold monomer models have a single model/chain\n",
    "\n",
    "print(f\"Extracted pLDDT-derived colors for {len(residue_colors):,} residues\")"
))

new_cells.append(md(
    "### 9c. Render the Structure\n",
    "Cartoon colored by confidence bin, with pathogenic (red) and benign (green) variant residues highlighted as sticks."
))

new_cells.append(code(
    "view = py3Dmol.view(width=900, height=600)\n",
    "view.addModel(pdb_str, \"pdb\")\n",
    "\n",
    "# Color the cartoon residue-by-residue using the pLDDT bins extracted above\n",
    "for resi, color in residue_colors.items():\n",
    "    view.addStyle({\"resi\": str(resi)}, {\"cartoon\": {\"color\": color}})\n",
    "\n",
    "# Overlay variant residues as sticks: red = pathogenic, green = benign\n",
    "for _, row in example_variants.iterrows():\n",
    "    color = \"red\" if row[\"Target\"] == 1 else \"green\"\n",
    "    view.addStyle({\"resi\": str(int(row[\"Pos\"]))}, {\"stick\": {\"color\": color, \"radius\": 0.4}})\n",
    "\n",
    "view.zoomTo()\n",
    "view.show()\n",
    "\n",
    "print(f\"\\n{EXAMPLE_UNIPROT_ID} (BRCA1) — AlphaFold confidence coloring (blue=very high \"\n",
    "      \"... orange=very low) with ClinVar variants overlaid (red=pathogenic, green=benign)\")"
))

nb["cells"].extend(new_cells)

with open(nb_path, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Appended {len(new_cells)} cells to {nb_path} (existing cells/outputs untouched)")
