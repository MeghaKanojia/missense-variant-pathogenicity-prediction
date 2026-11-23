import json

def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": list(lines)}

def code(*lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": list(lines)}

cells = []

# ============================================================ 0. Overview
cells.append(md(
    "# Phase 3: Graph Construction — From Structures to PAE-Weighted Protein Graphs\n",
    "This notebook turns each protein's AlphaFold structure into a graph the GNN can learn from, and attaches each "
    "ClinVar variant as a labelled training sample on that graph.\n\n"
    "**In one sentence:** each protein becomes a network where every amino acid is a node, two amino acids are "
    "connected if they sit close together in the folded 3D structure, and each connection is weighted by how *confident* "
    "AlphaFold is about that pair's relative position (the PAE-weighting that is this project's core novel idea).\n\n"
    "**Inputs:** `clinvar_with_structures.csv`, `alphafold_data/pdb/*.pdb`, `alphafold_data/pae/*.json`, "
    "`alphafold_data/metadata/*.json` (all from Phase 2)\n\n"
    "**Outputs:**\n"
    "- `esm2_embeddings/{UniProt_ID}.npy` — per-protein evolutionary embeddings (one row per residue)\n"
    "- `protein_graphs/{UniProt_ID}.pt` — one cached structural graph per protein (shared by all its variants)\n"
    "- `phase3_graph_manifest.csv` — per-protein status log for this phase\n"
    "- `clinvar_graph_ready.csv` — final variant table, restricted to variants whose protein graph built successfully\n"
    "- A `MissenseVariantGraphDataset` class (PyTorch Geometric `Dataset`) that turns each row of that table into a "
    "labelled training graph on demand — this is the object Phase 4 (model training) will load directly.\n\n"
    "**Key sources:**\n"
    "- Jumper, J. et al. (2021). *Highly accurate protein structure prediction with AlphaFold.* Nature, 596, 583–589. "
    "[doi:10.1038/s41586-021-03819-2](https://doi.org/10.1038/s41586-021-03819-2) — defines the Predicted Aligned Error "
    "(PAE) we use as edge weights.\n"
    "- Lin, Z. et al. (2023). *Evolutionary-scale prediction of atomic-level protein structure with a language model.* "
    "Science, 379(6637), 1123–1130. [doi:10.1126/science.ade2574](https://doi.org/10.1126/science.ade2574) — the ESM-2 "
    "protein language model used for node features.\n"
    "- Gligorijević, V. et al. (2021). *Structure-based protein function prediction using graph convolutional networks.* "
    "Nature Communications, 12, 3168. [doi:10.1038/s41467-021-23303-9](https://doi.org/10.1038/s41467-021-23303-9) — "
    "representative use of a Cα distance cutoff to define a residue contact graph, the convention we follow for edges.\n"
    "- Fey, M. & Lenssen, J.E. (2019). *Fast Graph Representation Learning with PyTorch Geometric.* ICLR Workshop on "
    "Representation Learning on Graphs and Manifolds. [arXiv:1903.02428](https://arxiv.org/abs/1903.02428) — the graph "
    "deep learning library we build on."
))

cells.append(code(
    "import os\n",
    "import json\n",
    "import gc\n",
    "from collections import OrderedDict\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import torch\n",
    "from torch_geometric.data import Data, Dataset\n",
    "from Bio.PDB import PDBParser\n",
    "import warnings\n",
    "from Bio import BiopythonWarning\n",
    "from tqdm.notebook import tqdm\n",
    "warnings.simplefilter(\"ignore\", BiopythonWarning)\n",
    "\n",
    "DEVICE = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
    "print(f\"Using device: {DEVICE}\")\n",
    "if DEVICE == \"cuda\":\n",
    "    print(f\"GPU: {torch.cuda.get_device_name(0)}\")"
))

# ============================================================ 1. Load Phase 2 output
cells.append(md(
    "## 1. Load Phase 2 Output\n",
    "We work from `clinvar_with_structures.csv` — every variant here already has a confirmed AlphaFold structure and "
    "PAE matrix (Phase 2), and has passed two independent reference-amino-acid checks (Phase 1 + Phase 2). Phase 3 adds "
    "a *third* layer of validation specific to graph-building, in Section 4, because it works directly off the raw "
    "structure/PAE files rather than their summary metadata."
))

cells.append(code(
    "df = pd.read_csv(\"clinvar_with_structures.csv\")\n",
    "print(f\"Variants carried into Phase 3 : {len(df):,}\")\n",
    "print(f\"Unique proteins                : {df['UniProt_ID'].nunique():,}\")\n",
    "print(f\"  Pathogenic (1): {(df['Target']==1).sum():,}\")\n",
    "print(f\"  Benign     (0): {(df['Target']==0).sum():,}\")\n",
    "\n",
    "BASE_DIR = \"alphafold_data\"\n",
    "PDB_DIR = os.path.join(BASE_DIR, \"pdb\")\n",
    "PAE_DIR = os.path.join(BASE_DIR, \"pae\")\n",
    "META_DIR = os.path.join(BASE_DIR, \"metadata\")\n",
    "\n",
    "ESM2_DIR = \"esm2_embeddings\"\n",
    "GRAPH_DIR = \"protein_graphs\"\n",
    "FIGURES_DIR = \"figures\"\n",
    "os.makedirs(ESM2_DIR, exist_ok=True)\n",
    "os.makedirs(GRAPH_DIR, exist_ok=True)\n",
    "os.makedirs(FIGURES_DIR, exist_ok=True)\n",
    "\n",
    "unique_proteins = sorted(df[\"UniProt_ID\"].unique().tolist())"
))

# ============================================================ 2. ESM-2 explanation
cells.append(md(
    "## 2. Node Features, Part 1: ESM-2 Evolutionary Embeddings\n",
    "### What ESM-2 is, in plain terms\n"
    "ESM-2 is a language model — the same kind of architecture as a text-generation AI — but instead of being trained "
    "on English sentences, it was trained on millions of real protein sequences from across all known life. By learning "
    "to predict 'what amino acid usually appears here, given the rest of the sequence', it implicitly learns which "
    "positions in a protein are evolutionarily constrained (changing them is usually harmful and rarely tolerated "
    "across species) versus tolerant of substitution (changing them is usually harmless). For every amino acid, ESM-2 "
    "outputs a vector of numbers (an *embedding*) summarising this learned context — we use this as a primary feature "
    "for every node in our graph.\n\n"
    "### Why include this at all, given we already have the 3D structure?\n"
    "The 3D structure tells the GNN about a position's *spatial* role (what's physically nearby). ESM-2 tells it about "
    "a position's *evolutionary* role (how tolerant this exact spot has historically been to change, based on patterns "
    "across millions of related sequences). These are complementary signals, and combining structure-based and "
    "sequence/evolutionary-based features is standard practice in modern variant-effect predictors, including "
    "AlphaMissense (Cheng et al., 2023, [doi:10.1126/science.adg7492](https://doi.org/10.1126/science.adg7492)) and "
    "gMVP (Ye et al., 2023, [doi:10.1038/s41467-023-42940-8](https://doi.org/10.1038/s41467-023-42940-8)).\n\n"
    "### Model choice\n"
    "We use `facebook/esm2_t33_650M_UR50D` — 650 million parameters, 33 transformer layers, producing a 1280-dimensional "
    "embedding per residue. This is a mid-sized ESM-2 checkpoint: large enough to capture rich evolutionary signal, "
    "small enough to run on a single consumer GPU (RTX 4050, 6 GB VRAM) one protein at a time."
))

cells.append(code(
    "from transformers import AutoTokenizer, AutoModel\n",
    "\n",
    "ESM2_MODEL_NAME = \"facebook/esm2_t33_650M_UR50D\"\n",
    "ESM2_EMBEDDING_DIM = 1280\n",
    "\n",
    "print(f\"Loading {ESM2_MODEL_NAME}...\")\n",
    "esm2_tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL_NAME)\n",
    "esm2_model = AutoModel.from_pretrained(ESM2_MODEL_NAME)\n",
    "esm2_model.eval()\n",
    "esm2_model.to(DEVICE)\n",
    "print(\"ESM-2 ready.\")"
))

cells.append(md(
    "## 3. Generate & Cache ESM-2 Embeddings (One Run Per Unique Protein)\n",
    "Each protein's sequence is run through ESM-2 exactly **once**, regardless of how many variants it has — embeddings "
    "are cached to disk (`esm2_embeddings/{UniProt_ID}.npy`) and reused for every variant on that protein.\n\n"
    "We use the canonical sequence cached in Phase 2's metadata (`uniprotSequence`), i.e. the sequence AlphaFold itself "
    "modelled — not a fresh UniProt fetch — to guarantee the embedding's positions line up exactly with the structure's "
    "residue numbering.\n\n"
    "**GPU memory handling:** ESM-2's memory use grows roughly with the square of sequence length (self-attention). We "
    "process one protein at a time in half precision (fp16) and fall back to CPU for any single protein that runs out "
    "of GPU memory, rather than letting one large protein crash the whole batch.\n\n"
    "**Validation:** after stripping the model's special start/end tokens, the number of embedding rows must exactly "
    "equal the protein's sequence length. Any mismatch is logged and that protein is excluded — this is a hard "
    "requirement, since a one-position offset here would silently scramble every node's identity in the graph."
))

cells.append(code(
    "def get_esm2_embedding(sequence, uniprot_id):\n",
    "    \"\"\"\n",
    "    Returns (embedding_array, status) where embedding_array has shape (len(sequence), 1280),\n",
    "    or (None, status) on failure. status in: 'cached', 'success', 'length_mismatch', 'failed'\n",
    "    \"\"\"\n",
    "    cache_path = os.path.join(ESM2_DIR, f\"{uniprot_id}.npy\")\n",
    "    if os.path.exists(cache_path):\n",
    "        return np.load(cache_path), \"cached\"\n",
    "\n",
    "    try:\n",
    "        inputs = esm2_tokenizer(sequence, return_tensors=\"pt\").to(DEVICE)\n",
    "        with torch.no_grad():\n",
    "            with torch.autocast(device_type=DEVICE if DEVICE == \"cuda\" else \"cpu\", dtype=torch.float16, enabled=(DEVICE == \"cuda\")):\n",
    "                out = esm2_model(**inputs)\n",
    "        # Strip <cls> (first) and <eos> (last) tokens to get one row per real residue\n",
    "        emb = out.last_hidden_state[0, 1:-1, :].float().cpu().numpy()\n",
    "    except torch.cuda.OutOfMemoryError:\n",
    "        torch.cuda.empty_cache()\n",
    "        try:\n",
    "            inputs = esm2_tokenizer(sequence, return_tensors=\"pt\")  # CPU fallback\n",
    "            with torch.no_grad():\n",
    "                out = esm2_model.to(\"cpu\")(**inputs)\n",
    "            emb = out.last_hidden_state[0, 1:-1, :].float().numpy()\n",
    "            esm2_model.to(DEVICE)  # restore for next protein\n",
    "        except Exception:\n",
    "            return None, \"failed\"\n",
    "    except Exception:\n",
    "        return None, \"failed\"\n",
    "\n",
    "    if emb.shape[0] != len(sequence):\n",
    "        return None, \"length_mismatch\"\n",
    "\n",
    "    np.save(cache_path, emb)\n",
    "    return emb, \"success\""
))

cells.append(code(
    "esm2_status = {}\n",
    "sequence_cache = {}\n",
    "\n",
    "for uid in tqdm(unique_proteins, desc=\"ESM-2 embeddings\"):\n",
    "    meta_path = os.path.join(META_DIR, f\"{uid}.json\")\n",
    "    if not os.path.exists(meta_path):\n",
    "        esm2_status[uid] = \"missing_metadata\"\n",
    "        continue\n",
    "    with open(meta_path, \"r\") as f:\n",
    "        entry = json.load(f)\n",
    "    seq = entry[\"uniprotSequence\"]\n",
    "    sequence_cache[uid] = seq\n",
    "\n",
    "    _, status = get_esm2_embedding(seq, uid)\n",
    "    esm2_status[uid] = status\n",
    "\n",
    "    if DEVICE == \"cuda\" and len(seq) > 1000:\n",
    "        torch.cuda.empty_cache()  # be conservative with VRAM for large proteins\n",
    "\n",
    "esm2_status_series = pd.Series(esm2_status)\n",
    "print(\"\\n=== ESM-2 embedding status ===\")\n",
    "print(esm2_status_series.value_counts().to_string())\n",
    "\n",
    "# Free GPU memory — ESM-2 is not needed again after this point\n",
    "del esm2_model\n",
    "gc.collect()\n",
    "if DEVICE == \"cuda\":\n",
    "    torch.cuda.empty_cache()"
))

# ============================================================ 4. Structural graph concept
cells.append(md(
    "## 4. Node Features, Part 2 & Edges: The Structural Contact Graph\n",
    "### Turning a 3D structure into a graph\n"
    "We treat each amino acid (residue) as a **node**. Two nodes get an **edge** between them if their Cα atoms — the "
    "central 'backbone' atom of each amino acid, standard for representing residue position — are within **8 Å** of "
    "each other in the folded 3D structure. This distance-cutoff contact graph is a widely used convention for "
    "representing protein structure as a graph for deep learning (e.g. Gligorijević et al., 2021, "
    "[doi:10.1038/s41467-021-23303-9](https://doi.org/10.1038/s41467-021-23303-9)). Intuitively: two amino acids that "
    "are close together in the folded protein can physically interact (e.g. forming part of the same binding pocket or "
    "structural core) even if they are far apart in the linear sequence — this is exactly the information a graph adds "
    "beyond a flat sequence.\n\n"
    "### Why weight edges by PAE — the project's central novel idea\n"
    "A binary 'are they within 8Å, yes or no' edge treats AlphaFold's prediction as ground truth with uniform "
    "confidence everywhere. In reality, AlphaFold reports, for every *pair* of residues, a **Predicted Aligned Error "
    "(PAE)** — its own estimate (in Ångströms) of how wrong the relative position of that pair might be (Jumper et al., "
    "2021). A low PAE means 'I am confident these two residues are positioned correctly relative to each other'; a high "
    "PAE means 'treat this pair's relative position with caution — often this happens between separate domains that "
    "move relative to one another, or in flexible linker regions'.\n\n"
    "Existing structure-based variant-effect GNNs (e.g. gMVP) use unweighted contact graphs — every edge counts "
    "equally. We instead convert each pair's PAE into an edge weight:\n\n"
    "```\n"
    "edge_weight = 1 / (1 + average(PAE[i][j], PAE[j][i]))\n"
    "```\n\n"
    "PAE is averaged across both directions because AlphaFold's PAE matrix is not symmetric (the error estimate from "
    "residue i's frame looking at j is not identical to j looking at i). The `1/(1+x)` transform maps PAE (0 Å = '"
    "perfectly confident' upward, unbounded) onto a bounded (0, 1] weight where 1 = maximum confidence and the weight "
    "shrinks smoothly as uncertainty grows. This lets the GNN learn to down-weight messages passed along structurally "
    "uncertain connections, rather than trusting every contact in the structure equally — this PAE-weighting is what "
    "distinguishes our approach from a standard contact-graph GNN.\n\n"
    "### Per-protein validation before building any graph\n"
    "Before constructing a graph, we require **four independent counts to agree exactly**: the number of residues with "
    "a Cα atom in the PDB file, the length of the canonical sequence from Phase 2's metadata, the dimension of the PAE "
    "matrix, and the length of the cached ESM-2 embedding. If any of these disagree, the protein is excluded rather than "
    "guessed at — a silent off-by-one here would misassign every node's identity and corrupt every variant built on "
    "that protein's graph."
))

cells.append(code(
    "DISTANCE_CUTOFF_ANGSTROM = 8.0\n",
    "AA_3TO1 = {\n",
    "    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',\n",
    "    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'\n",
    "}\n",
    "AA_1TO_ONEHOT_IDX = {aa: i for i, aa in enumerate(sorted(AA_3TO1.values()))}\n",
    "N_AA = len(AA_1TO_ONEHOT_IDX)  # 20\n",
    "\n",
    "pdb_parser = PDBParser(QUIET=True)\n",
    "\n",
    "def extract_ca_coords_and_sequence(uniprot_id):\n",
    "    \"\"\"Returns (ca_coords: dict[resnum -> xyz], seq_from_pdb: str) ordered by residue number.\"\"\"\n",
    "    pdb_path = os.path.join(PDB_DIR, f\"{uniprot_id}.pdb\")\n",
    "    structure = pdb_parser.get_structure(uniprot_id, pdb_path)\n",
    "    ca_coords = {}\n",
    "    resnames = {}\n",
    "    for model in structure:\n",
    "        for chain in model:\n",
    "            for residue in chain:\n",
    "                if residue.get_id()[0] != ' ':\n",
    "                    continue  # skip heteroatoms/waters\n",
    "                ca_atoms = [a for a in residue if a.get_id() == 'CA']\n",
    "                if ca_atoms:\n",
    "                    resnum = residue.get_id()[1]\n",
    "                    ca_coords[resnum] = ca_atoms[0].get_coord()\n",
    "                    resnames[resnum] = AA_3TO1.get(residue.get_resname(), 'X')\n",
    "        break  # single-chain monomer models\n",
    "    return ca_coords, resnames"
))

cells.append(code(
    "def build_protein_graph(uniprot_id):\n",
    "    \"\"\"\n",
    "    Builds and caches one structural graph for a protein.\n",
    "    Returns status string: 'cached', 'success', or a specific failure reason.\n",
    "    \"\"\"\n",
    "    graph_path = os.path.join(GRAPH_DIR, f\"{uniprot_id}.pt\")\n",
    "    if os.path.exists(graph_path):\n",
    "        return \"cached\"\n",
    "\n",
    "    if esm2_status.get(uniprot_id) not in (\"cached\", \"success\"):\n",
    "        return \"no_esm2_embedding\"\n",
    "\n",
    "    pae_path = os.path.join(PAE_DIR, f\"{uniprot_id}.json\")\n",
    "    meta_path = os.path.join(META_DIR, f\"{uniprot_id}.json\")\n",
    "    pdb_path = os.path.join(PDB_DIR, f\"{uniprot_id}.pdb\")\n",
    "    if not (os.path.exists(pae_path) and os.path.exists(meta_path) and os.path.exists(pdb_path)):\n",
    "        return \"missing_input_file\"\n",
    "\n",
    "    with open(meta_path, \"r\") as f:\n",
    "        canonical_seq = json.load(f)[\"uniprotSequence\"]\n",
    "    with open(pae_path, \"r\") as f:\n",
    "        pae_matrix = np.array(json.load(f)[0][\"predicted_aligned_error\"], dtype=np.float32)\n",
    "\n",
    "    ca_coords, resnames_from_pdb = extract_ca_coords_and_sequence(uniprot_id)\n",
    "    esm2_emb = np.load(os.path.join(ESM2_DIR, f\"{uniprot_id}.npy\"))\n",
    "\n",
    "    # --- 4-way consistency check: PDB residues == canonical sequence == PAE matrix == ESM-2 embedding ---\n",
    "    n_pdb = len(ca_coords)\n",
    "    n_seq = len(canonical_seq)\n",
    "    n_pae = pae_matrix.shape[0]\n",
    "    n_esm = esm2_emb.shape[0]\n",
    "    if not (n_pdb == n_seq == n_pae == n_esm):\n",
    "        return f\"length_mismatch (pdb={n_pdb}, seq={n_seq}, pae={n_pae}, esm2={n_esm})\"\n",
    "    if set(ca_coords.keys()) != set(range(1, n_pdb + 1)):\n",
    "        return \"non_contiguous_residue_numbering\"\n",
    "\n",
    "    # --- Build edges: Cα-Cα distance < cutoff ---\n",
    "    coords = np.stack([ca_coords[i] for i in range(1, n_pdb + 1)])  # shape (N, 3), ordered by residue number\n",
    "    diff = coords[:, None, :] - coords[None, :, :]\n",
    "    dist_matrix = np.sqrt((diff ** 2).sum(-1))\n",
    "    np.fill_diagonal(dist_matrix, np.inf)  # no self-loops\n",
    "    contact_mask = dist_matrix < DISTANCE_CUTOFF_ANGSTROM\n",
    "    src, dst = np.where(contact_mask)\n",
    "\n",
    "    distances = dist_matrix[src, dst]\n",
    "    pae_sym = (pae_matrix[src, dst] + pae_matrix[dst, src]) / 2.0\n",
    "    pae_weight = 1.0 / (1.0 + pae_sym)\n",
    "\n",
    "    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)\n",
    "    edge_attr = torch.tensor(np.stack([distances, pae_weight], axis=1), dtype=torch.float)  # [distance, pae_weight]\n",
    "\n",
    "    # --- Node features: ESM-2 embedding (1280) + wild-type AA one-hot (20) + 21 zero placeholder columns ---\n",
    "    # The 21 placeholder columns (1 variant-flag + 20 alt-AA one-hot) are filled in per-variant at load time\n",
    "    # by MissenseVariantGraphDataset (Section 6) — they stay zero in this shared base graph.\n",
    "    wt_onehot = np.zeros((n_pdb, N_AA), dtype=np.float32)\n",
    "    for i in range(n_pdb):\n",
    "        aa = canonical_seq[i]\n",
    "        if aa in AA_1TO_ONEHOT_IDX:\n",
    "            wt_onehot[i, AA_1TO_ONEHOT_IDX[aa]] = 1.0\n",
    "    placeholder = np.zeros((n_pdb, 1 + N_AA), dtype=np.float32)\n",
    "    x = np.concatenate([esm2_emb, wt_onehot, placeholder], axis=1)\n",
    "\n",
    "    data = Data(\n",
    "        x=torch.tensor(x, dtype=torch.float),\n",
    "        edge_index=edge_index,\n",
    "        edge_attr=edge_attr,\n",
    "        uniprot_id=uniprot_id,\n",
    "        num_residues=n_pdb,\n",
    "    )\n",
    "    torch.save(data, graph_path)\n",
    "    return \"success\""
))

cells.append(md(
    "## 5. Run Batch Graph Construction"
))

cells.append(code(
    "graph_status = {}\n",
    "for uid in tqdm(unique_proteins, desc=\"Building protein graphs\"):\n",
    "    graph_status[uid] = build_protein_graph(uid)\n",
    "\n",
    "graph_status_series = pd.Series(graph_status)\n",
    "print(\"\\n=== Protein graph construction status ===\")\n",
    "print(graph_status_series.value_counts().to_string())"
))

# ============================================================ 6. Variant-level dataset
cells.append(md(
    "## 6. From Protein Graphs to Labelled Variant Samples\n",
    "### Why one shared graph per protein, not one graph per variant\n"
    "A protein's structure and edges never change between its variants — only the GNN's *input signal at one node* "
    "changes, the location and identity of the substitution. Saving a full duplicate graph per variant would multiply "
    "disk usage by the number of variants per protein for no benefit (BRCA1 alone has 318 variants sharing one "
    "1,863-residue structure). Instead we cache the structure once (Section 5) and apply two small, cheap edits to a "
    "copy of its node features at load time:\n\n"
    "1. **Variant-site flag** — set to `1` only at the mutated residue's node, `0` everywhere else. This tells the GNN "
    "*where* to focus.\n"
    "2. **Alt-AA one-hot** — a 20-length one-hot vector identifying *what* the residue was substituted to, written "
    "only at the mutated node (zero everywhere else).\n\n"
    "The wild-type identity and ESM-2 embedding at that same node are left untouched, so the GNN sees: 'this is "
    "ordinarily residue X here (per the embedding and one-hot), in this sample it has become residue Y, here is the "
    "rest of the protein's structural and evolutionary context unchanged.'\n\n"
    "### Bounding memory use\n"
    "Across our ~6,600 proteins (mean length 715 residues), caching every distinct protein's graph in memory "
    "simultaneously would need roughly 25 GB for node features alone — more than this machine has free. "
    "`MissenseVariantGraphDataset` therefore keeps only the most-recently-used 64 protein graphs in memory (a bounded "
    "LRU cache), evicting the oldest once that limit is reached, rather than letting the cache grow without bound over "
    "a full pass through the dataset.\n\n"
    "### Implementation\n"
    "`MissenseVariantGraphDataset` is a standard PyTorch Geometric `Dataset` (Fey & Lenssen, 2019). Each item corresponds "
    "to one row of `clinvar_graph_ready.csv` (one variant). `get(idx)` loads the cached base graph for that variant's "
    "protein, clones its node feature tensor, writes the two edits above at the variant's position, and attaches the "
    "ClinVar label (`Target`) as `y`. This means the heavy structural data is read from disk once per access but never "
    "duplicated — exactly the same approach used when, e.g., loading a shared background image for many separate "
    "annotated crops in computer vision."
))

cells.append(code(
    "class MissenseVariantGraphDataset(Dataset):\n",
    "    \"\"\"\n",
    "    Each item is one ClinVar missense variant, represented as the shared structural graph for its protein\n",
    "    with two node-level edits applied at the variant's position: a variant-site flag and the substituted\n",
    "    (Alt_AA) one-hot identity. Label (y) is the ClinVar pathogenicity Target (1 = pathogenic, 0 = benign).\n",
    "    \"\"\"\n",
    "\n",
    "    def __init__(self, variant_df, graph_dir, cache_size=64):\n",
    "        super().__init__()\n",
    "        self.variant_df = variant_df.reset_index(drop=True)\n",
    "        self.graph_dir = graph_dir\n",
    "        self.cache_size = cache_size\n",
    "        # Bounded LRU cache, not an unbounded dict: across our ~6,600 proteins, node features alone would\n",
    "        # total ~25 GB if every distinct protein touched in one pass were cached simultaneously (mean protein\n",
    "        # length 715 residues x 1321 features x 4 bytes x 6,604 proteins) - far more than typical machine RAM.\n",
    "        # Keeping only the most-recently-used `cache_size` proteins in memory bounds this to a few hundred MB.\n",
    "        self._graph_cache = OrderedDict()\n",
    "\n",
    "    def len(self):\n",
    "        return len(self.variant_df)\n",
    "\n",
    "    def _load_base_graph(self, uniprot_id):\n",
    "        if uniprot_id in self._graph_cache:\n",
    "            self._graph_cache.move_to_end(uniprot_id)\n",
    "            return self._graph_cache[uniprot_id]\n",
    "        graph = torch.load(os.path.join(self.graph_dir, f\"{uniprot_id}.pt\"), weights_only=False)\n",
    "        self._graph_cache[uniprot_id] = graph\n",
    "        if len(self._graph_cache) > self.cache_size:\n",
    "            self._graph_cache.popitem(last=False)  # evict least-recently-used\n",
    "        return graph\n",
    "\n",
    "    def get(self, idx):\n",
    "        row = self.variant_df.iloc[idx]\n",
    "        base_graph = self._load_base_graph(row[\"UniProt_ID\"])\n",
    "\n",
    "        x = base_graph.x.clone()\n",
    "        node_idx = int(row[\"Pos\"]) - 1  # PDB/UniProt positions are 1-indexed; tensor rows are 0-indexed\n",
    "\n",
    "        VARIANT_FLAG_COL = ESM2_EMBEDDING_DIM + N_AA          # column 1300\n",
    "        ALT_AA_START_COL = VARIANT_FLAG_COL + 1                # columns 1301..1320\n",
    "\n",
    "        x[node_idx, VARIANT_FLAG_COL] = 1.0\n",
    "        alt_idx = AA_1TO_ONEHOT_IDX[row[\"Alt_AA_1\"]]\n",
    "        x[node_idx, ALT_AA_START_COL + alt_idx] = 1.0\n",
    "\n",
    "        return Data(\n",
    "            x=x,\n",
    "            edge_index=base_graph.edge_index,\n",
    "            edge_attr=base_graph.edge_attr,\n",
    "            y=torch.tensor([row[\"Target\"]], dtype=torch.long),\n",
    "            uniprot_id=row[\"UniProt_ID\"],\n",
    "            variation_id=row[\"VariationID\"],\n",
    "            variant_pos=int(row[\"Pos\"]),\n",
    "        )"
))

# ============================================================ 7. Validation & manifest
cells.append(md(
    "## 7. Final Validation, Manifest & Variant Table\n",
    "A variant proceeds to `clinvar_graph_ready.csv` only if its protein's graph was built successfully. We also "
    "convert `Alt_AA` (3-letter code, e.g. `Arg`) to the 1-letter code used by the one-hot encoder, and verify it is one "
    "of the 20 standard amino acids (ClinVar occasionally encodes unusual substitutions that we should not silently "
    "miscode)."
))

cells.append(code(
    "manifest_df = pd.DataFrame({\n",
    "    \"UniProt_ID\": unique_proteins,\n",
    "    \"esm2_status\": [esm2_status.get(u) for u in unique_proteins],\n",
    "    \"graph_status\": [graph_status.get(u) for u in unique_proteins],\n",
    "})\n",
    "manifest_df[\"graph_ok\"] = manifest_df[\"graph_status\"].isin([\"cached\", \"success\"])\n",
    "manifest_df.to_csv(\"phase3_graph_manifest.csv\", index=False)\n",
    "print(f\"Proteins with a usable graph: {manifest_df['graph_ok'].sum():,} / {len(manifest_df):,}\")\n",
    "\n",
    "AA_3TO1_UPPER_KEYS = {k: v for k, v in AA_3TO1.items()}\n",
    "def three_to_one(code3):\n",
    "    return AA_3TO1_UPPER_KEYS.get(str(code3).upper())\n",
    "\n",
    "df[\"Alt_AA_1\"] = df[\"Alt_AA\"].apply(three_to_one)\n",
    "\n",
    "graph_ok_proteins = set(manifest_df.loc[manifest_df[\"graph_ok\"], \"UniProt_ID\"])\n",
    "before = len(df)\n",
    "graph_ready_df = df[\n",
    "    df[\"UniProt_ID\"].isin(graph_ok_proteins) & df[\"Alt_AA_1\"].notna()\n",
    "].copy()\n",
    "\n",
    "print(f\"\\nVariants before Phase 3 filter : {before:,}\")\n",
    "print(f\"Variants after Phase 3 filter  : {len(graph_ready_df):,} (dropped {before - len(graph_ready_df):,})\")\n",
    "print(f\"  Pathogenic (1): {(graph_ready_df['Target']==1).sum():,}\")\n",
    "print(f\"  Benign     (0): {(graph_ready_df['Target']==0).sum():,}\")\n",
    "print(f\"Unique proteins : {graph_ready_df['UniProt_ID'].nunique():,}\")\n",
    "\n",
    "graph_ready_df.to_csv(\"clinvar_graph_ready.csv\", index=False)\n",
    "print(\"\\nSaved clinvar_graph_ready.csv — input for Phase 4 (model training)\")"
))

cells.append(md(
    "## 7b. Final Per-Variant Consistency Check: Ref_AA vs. the Literal Graph Node\n",
    "Phases 1 and 2 already validated `Ref_AA` against the live UniProt sequence and AlphaFold's modelled sequence "
    "respectively. This final check closes the loop by re-reading the *actual cached graph file* each variant will be "
    "served from in Phase 4, and confirming the wild-type one-hot block at that variant's node still encodes `Ref_AA`. "
    "This catches any error introduced specifically during this notebook's own graph-building step (e.g. an indexing "
    "slip when assembling `x`) — a class of bug the earlier phases cannot see, since they never touch the graph "
    "tensors themselves. We expect zero mismatches; any that appear are dropped defensively and reported rather than "
    "silently passed through to model training."
))

cells.append(code(
    "print(\"Re-validating Ref_AA against the literal node feature stored in each protein's cached graph...\")\n",
    "mismatches = []\n",
    "for uid, group in tqdm(graph_ready_df.groupby(\"UniProt_ID\"), desc=\"Final graph consistency check\"):\n",
    "    graph = torch.load(os.path.join(GRAPH_DIR, f\"{uid}.pt\"), weights_only=False)\n",
    "    wt_block = graph.x[:, ESM2_EMBEDDING_DIM:ESM2_EMBEDDING_DIM + N_AA].numpy()\n",
    "    for _, row in group.iterrows():\n",
    "        node_idx = int(row[\"Pos\"]) - 1\n",
    "        ref_1 = AA_3TO1.get(str(row[\"Ref_AA\"]).upper())\n",
    "        ref_idx = AA_1TO_ONEHOT_IDX.get(ref_1) if ref_1 else None\n",
    "        if (\n",
    "            ref_idx is None\n",
    "            or node_idx < 0\n",
    "            or node_idx >= wt_block.shape[0]\n",
    "            or wt_block[node_idx, ref_idx] != 1.0\n",
    "        ):\n",
    "            mismatches.append((row[\"VariationID\"], uid, row[\"Pos\"], row[\"Ref_AA\"]))\n",
    "\n",
    "print(f\"Checked {len(graph_ready_df):,} variants directly against their graph's node features.\")\n",
    "print(f\"Mismatches found: {len(mismatches)}\")\n",
    "\n",
    "if mismatches:\n",
    "    print(\"Dropping mismatched variants (unexpected — investigate if this list is non-empty):\")\n",
    "    print(mismatches[:10])\n",
    "    bad_ids = {m[0] for m in mismatches}\n",
    "    graph_ready_df = graph_ready_df[~graph_ready_df[\"VariationID\"].isin(bad_ids)].copy()\n",
    "    graph_ready_df.to_csv(\"clinvar_graph_ready.csv\", index=False)\n",
    "    print(f\"Re-saved clinvar_graph_ready.csv with {len(graph_ready_df):,} variants after dropping mismatches.\")\n",
    "else:\n",
    "    print(\"All variants confirmed consistent with their graph's literal node features. No drops needed.\")"
))

cells.append(md(
    "## 8. Sanity Check: Instantiate the Dataset and Inspect One Sample\n",
    "We build the dataset object and pull out one real variant to confirm the shapes and the variant-site edit are "
    "exactly as designed before moving to model training."
))

cells.append(code(
    "dataset = MissenseVariantGraphDataset(graph_ready_df, GRAPH_DIR)\n",
    "print(f\"Dataset size: {len(dataset):,} variants\\n\")\n",
    "\n",
    "sample = dataset[0]\n",
    "print(sample)\n",
    "print(f\"\\nNode feature dim         : {sample.x.shape[1]} (expected {ESM2_EMBEDDING_DIM + N_AA + 1 + N_AA})\")\n",
    "print(f\"Number of edges           : {sample.edge_index.shape[1]:,}\")\n",
    "print(f\"Label (Target)            : {sample.y.item()}\")\n",
    "\n",
    "node_idx = sample.variant_pos - 1\n",
    "variant_flag_col = ESM2_EMBEDDING_DIM + N_AA\n",
    "print(f\"\\nVariant-site flag at mutated node : {sample.x[node_idx, variant_flag_col].item()} (expected 1.0)\")\n",
    "other_node = 0 if node_idx != 0 else 1\n",
    "print(f\"Variant-site flag at a random other node : {sample.x[other_node, variant_flag_col].item()} (expected 0.0)\")"
))

cells.append(md(
    "## 9. Visual Sanity Check: Contact Map vs. PAE-Weight Map\n",
    "A quick visual cross-check on the same example protein used in Phase 2 (BRCA1, `P38398`): the binary contact map "
    "(which residue pairs are within 8 Å) next to the PAE-derived confidence weight for those same pairs. We expect the "
    "PAE weights to be high (bright) along the diagonal and in clearly folded local neighbourhoods, and lower in "
    "between distant domains — this is a useful figure for the methodology section, showing concretely what the "
    "'PAE-weighted' part of the architecture is operating on."
))

cells.append(code(
    "import matplotlib.pyplot as plt\n",
    "\n",
    "example_id = \"P38398\"\n",
    "g = torch.load(os.path.join(GRAPH_DIR, f\"{example_id}.pt\"), weights_only=False)\n",
    "n = g.num_residues\n",
    "\n",
    "adj = np.zeros((n, n))\n",
    "pae_w = np.zeros((n, n))\n",
    "src, dst = g.edge_index.numpy()\n",
    "adj[src, dst] = 1\n",
    "pae_w[src, dst] = g.edge_attr[:, 1].numpy()\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(13, 6))\n",
    "axes[0].imshow(adj, cmap=\"Greys\")\n",
    "axes[0].set_title(f\"{example_id}: Binary Contact Map (Cα < {DISTANCE_CUTOFF_ANGSTROM} A)\")\n",
    "axes[1].imshow(pae_w, cmap=\"viridis\")\n",
    "axes[1].set_title(f\"{example_id}: PAE-Derived Edge Weight (brighter = more confident)\")\n",
    "for ax in axes:\n",
    "    ax.set_xlabel(\"Residue index\")\n",
    "    ax.set_ylabel(\"Residue index\")\n",
    "plt.tight_layout()\n",
    "plt.show()"
))

cells.append(md(
    "## 10. Visualizing a Local Graph Neighborhood (For a Non-Specialist Audience)\n",
    "The full protein graph (1,863 nodes for BRCA1 alone) is too dense to draw in a way anyone could read. Instead, we "
    "zoom into the **immediate neighborhood of a single real variant** — the mutated residue itself, plus every "
    "residue directly connected to it (its 1-hop neighbors) — to show concretely what a 'node' and an 'edge' actually "
    "are in this project.\n\n"
    "**How to read this figure:**\n"
    "- Each circle is one amino acid (a node), labelled with its identity and position (e.g. `C64` = Cysteine at "
    "position 64).\n"
    "- The **red** circle is the variant position itself — this is the node the GNN will mark with the variant-site "
    "flag and substituted-residue identity described in Section 6.\n"
    "- A line between two circles (an edge) means those two residues are within 8 Å of each other in the real 3D fold "
    "— note several of these neighbors are far apart in the sequence (e.g. residue 39 connecting to residue 64), which "
    "is only visible because we used the *3D structure*, not just the linear sequence.\n"
    "- The **color and thickness** of each line is the PAE-derived confidence weight from Section 4 — brighter/thicker "
    "= AlphaFold is more confident about that pair's relative position; darker/thinner = less confident. This is the "
    "literal quantity the GNN's message-passing will weight by."
))

cells.append(code(
    "!pip install networkx -q"
))

cells.append(code(
    "import networkx as nx\n",
    "\n",
    "viz_uniprot_id = \"P38398\"  # BRCA1, consistent with earlier examples in this notebook\n",
    "g_viz = torch.load(os.path.join(GRAPH_DIR, f\"{viz_uniprot_id}.pt\"), weights_only=False)\n",
    "with open(os.path.join(META_DIR, f\"{viz_uniprot_id}.json\")) as f:\n",
    "    viz_seq = json.load(f)[\"uniprotSequence\"]\n",
    "\n",
    "viz_variants = graph_ready_df[graph_ready_df[\"UniProt_ID\"] == viz_uniprot_id]\n",
    "example_variant = viz_variants[viz_variants[\"Target\"] == 1].iloc[0]\n",
    "center_node = int(example_variant[\"Pos\"]) - 1\n",
    "\n",
    "src, dst = g_viz.edge_index.numpy()\n",
    "neighbors_1hop = (set(dst[src == center_node].tolist()) | set(src[dst == center_node].tolist())) - {center_node}\n",
    "subgraph_nodes = sorted({center_node} | neighbors_1hop)\n",
    "\n",
    "mask = np.isin(src, subgraph_nodes) & np.isin(dst, subgraph_nodes)\n",
    "sub_src, sub_dst, sub_weights = src[mask], dst[mask], g_viz.edge_attr[mask, 1].numpy()\n",
    "\n",
    "G = nx.Graph()\n",
    "for n in subgraph_nodes:\n",
    "    G.add_node(n, label=f\"{viz_seq[n]}{n+1}\")\n",
    "for s, d, w in zip(sub_src, sub_dst, sub_weights):\n",
    "    G.add_edge(int(s), int(d), weight=float(w))\n",
    "\n",
    "pos = nx.spring_layout(G, seed=42, k=0.8)\n",
    "node_colors = [\"#E63946\" if n == center_node else \"#A8DADC\" for n in G.nodes()]\n",
    "node_sizes = [1400 if n == center_node else 900 for n in G.nodes()]\n",
    "edge_weights_list = [G[u][v][\"weight\"] for u, v in G.edges()]\n",
    "edge_widths = [1 + 4 * w for w in edge_weights_list]\n",
    "\n",
    "plt.figure(figsize=(8, 7))\n",
    "nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_weights_list,\n",
    "                        edge_cmap=plt.cm.viridis, edge_vmin=0, edge_vmax=1)\n",
    "nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, edgecolors=\"black\")\n",
    "nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, \"label\"), font_size=10, font_weight=\"bold\")\n",
    "plt.title(\n",
    "    f\"{viz_uniprot_id} (BRCA1): Local Graph Neighborhood Around a Real Variant\\n\"\n",
    "    f\"{example_variant['Ref_AA']}{example_variant['Pos']}{example_variant['Alt_AA']} (Pathogenic)\",\n",
    "    fontsize=12\n",
    ")\n",
    "plt.axis(\"off\")\n",
    "sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=1))\n",
    "sm.set_array([])\n",
    "cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.7)\n",
    "cbar.set_label(\"Edge weight (PAE confidence)\", fontsize=9)\n",
    "plt.tight_layout()\n",
    "plt.show()"
))

cells.append(md(
    "## 11. Methods Summary (Copy-Paste for the Paper)\n",
    "> Each protein was represented as a graph in which nodes correspond to individual amino acid residues and edges "
    "connect residue pairs within 8 Å (Cα–Cα distance) in the AlphaFold2-predicted structure. Node features combined "
    "a 1280-dimensional ESM-2 protein language model embedding with a one-hot encoding of wild-type residue identity. "
    "Edges were weighted by AlphaFold's Predicted Aligned Error (PAE) between the residue pair, transformed as "
    "1/(1+PAE), so that the model could learn to discount structurally uncertain contacts rather than treating all "
    "spatial contacts as equally reliable — this PAE-weighting constitutes the central architectural novelty of this "
    "work relative to prior unweighted contact-graph approaches (e.g. gMVP). Each ClinVar missense variant was "
    "represented as an instance of its protein's shared structural graph, with the mutated node's feature vector "
    "augmented by a variant-site indicator and the substituted residue's one-hot identity, labelled with the variant's "
    "ClinVar pathogenicity classification.\n\n"
    "**Supporting numbers to cite alongside this paragraph (filled in once this notebook has been run in full):**\n"
    "- Cα distance cutoff: 8 Å\n"
    "- ESM-2 checkpoint: `facebook/esm2_t33_650M_UR50D` (650M parameters, 1280-dim output)\n"
    "- Node feature dimensionality: 1321 (1280 ESM-2 + 20 wild-type one-hot + 1 variant-site flag + 20 alt-residue "
    "one-hot)\n"
    "- See `phase3_graph_manifest.csv` for per-protein graph-construction success rate, and the variant counts printed "
    "in Section 7 for the final `clinvar_graph_ready.csv` totals."
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

out_path = r"c:\Users\Megha Kanojia\Documents\DevProjects\agy2-projects\missense-variants\phase3_graph_construction.ipynb"
with open(out_path, "w") as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook written to {out_path}")
