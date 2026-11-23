import json

def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": list(lines)}

def code(*lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": list(lines)}

cells = []

cells.append(md(
    "# Phase 2: AlphaFold Structure & PAE Asset Download\n",
    "This notebook downloads the AlphaFold2-predicted 3D structure (`.pdb`) and Predicted Aligned Error matrix (`.json`) "
    "for every unique UniProt accession produced by Phase 1 (`clinvar_missense_filtered.csv`, 6,684 unique proteins), "
    "and **independently re-verifies** that every variant's `(Ref_AA, Pos)` is consistent with the exact sequence "
    "AlphaFold actually modelled — not just the UniProt sequence we checked against in Phase 1.\n\n"
    "**Inputs:** `clinvar_missense_filtered.csv` (Phase 1 output)\n\n"
    "**Outputs:**\n"
    "- `alphafold_data/metadata/{UniProt_ID}.json` — cached AlphaFold prediction metadata per protein\n"
    "- `alphafold_data/pdb/{UniProt_ID}.pdb` — structure file per protein\n"
    "- `alphafold_data/pae/{UniProt_ID}.json` — PAE matrix per protein\n"
    "- `asset_download_manifest.csv` — per-protein download + structural-validation status\n"
    "- `clinvar_with_structures.csv` — Phase 1 variants restricted to proteins with a fully verified structure+PAE pair, for Phase 3\n\n"
    "**Primary sources:**\n"
    "- Varadi, M. et al. (2022). *AlphaFold Protein Structure Database: massively expanding the structural coverage of "
    "protein-sequence space with high-accuracy models.* Nucleic Acids Research, 50(D1), D439–D444. "
    "[doi:10.1093/nar/gkab1061](https://doi.org/10.1093/nar/gkab1061)\n"
    "- Jumper, J. et al. (2021). *Highly accurate protein structure prediction with AlphaFold.* Nature, 596, 583–589. "
    "[doi:10.1038/s41586-021-03819-2](https://doi.org/10.1038/s41586-021-03819-2)\n"
    "- Tunyasuvunakool, K. et al. (2021). *Highly accurate protein structure prediction for the human proteome.* Nature, "
    "596, 590–596. [doi:10.1038/s41586-021-03828-1](https://doi.org/10.1038/s41586-021-03828-1)"
))

cells.append(code(
    "import os\n",
    "import json\n",
    "import time\n",
    "import requests\n",
    "import pandas as pd\n",
    "from concurrent.futures import ThreadPoolExecutor, as_completed\n",
    "from tqdm.notebook import tqdm"
))

cells.append(md(
    "## 1. Load Phase 1 Output & Get Unique Proteins\n",
    "We only need to fetch one structure and one PAE file per unique `UniProt_ID` — multiple variants in the same protein "
    "share the same structure."
))

cells.append(code(
    "input_file = \"clinvar_missense_filtered.csv\"\n",
    "df = pd.read_csv(input_file)\n",
    "print(f\"Loaded {len(df):,} variants from Phase 1\")\n",
    "\n",
    "unique_uniprots = sorted(df['UniProt_ID'].unique().tolist())\n",
    "print(f\"Unique UniProt IDs to fetch: {len(unique_uniprots):,}\")"
))

cells.append(md(
    "## 2. Set Up Local Storage\n",
    "Metadata, structures, and PAE matrices are saved to disk under `alphafold_data/`, separated by file type. Every "
    "download step below skips files that already exist on disk, so the pipeline is idempotent and safely resumable "
    "after an interruption."
))

cells.append(code(
    "BASE_DIR = \"alphafold_data\"\n",
    "META_DIR = os.path.join(BASE_DIR, \"metadata\")\n",
    "PDB_DIR = os.path.join(BASE_DIR, \"pdb\")\n",
    "PAE_DIR = os.path.join(BASE_DIR, \"pae\")\n",
    "for d in (META_DIR, PDB_DIR, PAE_DIR):\n",
    "    os.makedirs(d, exist_ok=True)\n",
    "print(f\"Metadata : {os.path.abspath(META_DIR)}\")\n",
    "print(f\"PDB      : {os.path.abspath(PDB_DIR)}\")\n",
    "print(f\"PAE      : {os.path.abspath(PAE_DIR)}\")"
))

cells.append(md(
    "## 3. Fetch Official AlphaFold Prediction Metadata\n",
    "### Why metadata-driven download, not a hardcoded URL pattern\n"
    "A common (and fragile) shortcut is to hardcode the file URL pattern, e.g. "
    "`AF-{accession}-F1-model_v4.pdb`. This assumes every entry is on model version `v4` and is a single, unfragmented "
    "model — neither is guaranteed. Instead we query AlphaFold DB's **official prediction metadata API** "
    "(`GET https://alphafold.ebi.ac.uk/api/prediction/{accession}`, documented alongside the AlphaFold DB release, "
    "Varadi et al. 2022), which returns the *exact, current* download URLs (`pdbUrl`, `paeDocUrl`) together with the "
    "`uniprotSequence` actually used to build that specific model, its UniProt coordinate range (`uniprotStart`, "
    "`uniprotEnd`), and the released model version (`latestVersion`). Driving the download from this metadata, rather "
    "than from an assumed filename, is the approach the AlphaFold DB documentation itself recommends for programmatic "
    "access.\n\n"
    "### Why this matters for correctness, not just convenience\n"
    "This metadata is also what lets us catch a subtle but important failure mode in Section 6: **UniProt's canonical "
    "sequence can be revised after an AlphaFold model was built.** Phase 1 validated each variant's `Ref_AA` against "
    "*today's* UniProt FASTA. If UniProt's canonical sequence has since changed (isoform reassignment, sequence "
    "correction, etc.), the residue at `Pos` in the *actual AlphaFold structure* could differ from what Phase 1 verified, "
    "silently corrupting the GNN's structural input. `uniprotSequence` in this metadata is the ground truth for what the "
    "downloaded `.pdb` file actually represents, so re-checking against it closes that gap. This mirrors the coordinate "
    "sanity-checking practice used by gMVP (Ye et al., 2023, "
    "[doi:10.1038/s41467-023-42940-8](https://doi.org/10.1038/s41467-023-42940-8)) and AlphaMissense (Cheng et al., 2023, "
    "[doi:10.1126/science.adg7492](https://doi.org/10.1126/science.adg7492)) before structure-conditioned features are "
    "computed.\n\n"
    "### The API returns isoforms too — must filter to the canonical entry\n"
    "Empirically, querying this endpoint for a given accession returns **one entry per modelled isoform of that gene**, "
    "not just the canonical sequence — e.g. querying `P38398` (BRCA1) returns 8 entries: the canonical model "
    "(`uniprotAccession == \"P38398\"`) plus 7 alternate-isoform models (`uniprotAccession` `\"P38398-2\"`, "
    "`\"P38398-3\"`, ... `\"P38398-8\"`), each a different length. Phase 1's UniProt mapping and FASTA validation were "
    "performed against the **canonical** accession only, so we must explicitly select the entry whose `uniprotAccession` "
    "is an *exact* match to the queried ID (no isoform suffix) — naively taking `entries[0]` or treating "
    "`len(entries) > 1` as an error would silently misidentify or wrongly discard canonical proteins.\n\n"
    "### Fragmented proteins\n"
    "Separately, very large proteins can have the *canonical* accession itself split into overlapping structural "
    "fragments (multiple entries all with `uniprotAccession == uniprot_id`, but different `uniprotStart`/`uniprotEnd` "
    "ranges). Phase 1 already excluded proteins >2,700 AA — the threshold at which AlphaFold DB begins fragmenting — so "
    "we expect none of these, but we check explicitly (more than one canonical-accession entry, or `uniprotStart != 1`) "
    "rather than assuming Phase 1's length filter was sufficient."
))

cells.append(code(
    "ALPHAFOLD_API_BASE = \"https://alphafold.ebi.ac.uk/api/prediction\"\n",
    "MAX_RETRIES = 3\n",
    "RETRY_BACKOFF_SECONDS = 2\n",
    "\n",
    "def fetch_metadata(session, uniprot_id):\n",
    "    \"\"\"\n",
    "    Fetch (and cache) the official AlphaFold prediction metadata for one UniProt accession, filtered down to the\n",
    "    single canonical-isoform entry. Returns (canonical_entry_dict_or_None, status) where status is one of:\n",
    "    'cached', 'success', 'not_found', 'fragmented', 'failed'\n",
    "    \"\"\"\n",
    "    cache_path = os.path.join(META_DIR, f\"{uniprot_id}.json\")\n",
    "    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:\n",
    "        with open(cache_path, \"r\") as f:\n",
    "            entry = json.load(f)\n",
    "        return entry, \"cached\"\n",
    "\n",
    "    url = f\"{ALPHAFOLD_API_BASE}/{uniprot_id}\"\n",
    "    for attempt in range(1, MAX_RETRIES + 1):\n",
    "        try:\n",
    "            resp = session.get(url, timeout=30)\n",
    "            if resp.status_code == 200:\n",
    "                entries = resp.json()\n",
    "                if not entries:\n",
    "                    return None, \"not_found\"\n",
    "\n",
    "                # The API returns one entry per modelled isoform (e.g. P38398, P38398-2, ... P38398-8).\n",
    "                # We want only the canonical entry — an exact accession match, no isoform suffix.\n",
    "                canonical = [e for e in entries if e.get(\"uniprotAccession\") == uniprot_id]\n",
    "                if not canonical:\n",
    "                    return None, \"not_found\"\n",
    "                if len(canonical) > 1:\n",
    "                    # Genuine large-protein fragmentation: same canonical accession, multiple position ranges.\n",
    "                    return canonical, \"fragmented\"\n",
    "\n",
    "                entry = canonical[0]\n",
    "                if entry.get(\"uniprotStart\") != 1:\n",
    "                    return entry, \"fragmented\"\n",
    "\n",
    "                with open(cache_path, \"w\") as f:\n",
    "                    json.dump(entry, f)\n",
    "                return entry, \"success\"\n",
    "            elif resp.status_code == 404:\n",
    "                return None, \"not_found\"  # permanent — no model for this accession\n",
    "            else:\n",
    "                time.sleep(RETRY_BACKOFF_SECONDS * attempt)\n",
    "        except requests.exceptions.RequestException:\n",
    "            time.sleep(RETRY_BACKOFF_SECONDS * attempt)\n",
    "\n",
    "    return None, \"failed\""
))

cells.append(md(
    "## 4. Download Structure & PAE Files\n",
    "Using the exact `pdbUrl` / `paeDocUrl` returned by the metadata call above (not an assumed filename), we download "
    "both files. As in Section 3, we distinguish a permanent `404` (no asset at that URL — shouldn't occur once metadata "
    "succeeded, but checked defensively) from transient errors (timeout, `5xx`), which are retried with backoff. Files "
    "already present on disk are skipped."
))

cells.append(code(
    "def download_one(session, url, dest_path):\n",
    "    \"\"\"Download a single file with retry-with-backoff. Returns 'success', 'already_exists', 'not_found', or 'failed'.\"\"\"\n",
    "    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:\n",
    "        return \"already_exists\"\n",
    "\n",
    "    for attempt in range(1, MAX_RETRIES + 1):\n",
    "        try:\n",
    "            resp = session.get(url, timeout=30)\n",
    "            if resp.status_code == 200 and resp.content:\n",
    "                with open(dest_path, \"wb\") as f:\n",
    "                    f.write(resp.content)\n",
    "                return \"success\"\n",
    "            elif resp.status_code == 404:\n",
    "                return \"not_found\"\n",
    "            else:\n",
    "                time.sleep(RETRY_BACKOFF_SECONDS * attempt)\n",
    "        except requests.exceptions.RequestException:\n",
    "            time.sleep(RETRY_BACKOFF_SECONDS * attempt)\n",
    "\n",
    "    return \"failed\""
))

cells.append(md(
    "## 5. Run Batch Pipeline (Metadata + Download, per Protein)\n",
    "For each protein: fetch metadata (Section 3), then — only if metadata succeeded and the model is full-length and "
    "unfragmented — download the structure and PAE files using the URLs from that metadata. We parallelise across "
    "proteins with a small thread pool (8 workers); AlphaFold DB is served from an EMBL-EBI CDN, so this is fast without "
    "being an aggressive load on a shared public resource. Re-running this cell after an interruption only fetches what's "
    "still missing."
))

cells.append(code(
    "def process_protein(uniprot_id):\n",
    "    session = requests.Session()\n",
    "    metadata, meta_status = fetch_metadata(session, uniprot_id)\n",
    "\n",
    "    record = {\n",
    "        \"UniProt_ID\": uniprot_id,\n",
    "        \"meta_status\": meta_status,\n",
    "        \"pdb_status\": None,\n",
    "        \"pae_status\": None,\n",
    "        \"pdb_path\": None,\n",
    "        \"pae_path\": None,\n",
    "        \"uniprot_sequence\": None,\n",
    "        \"uniprot_start\": None,\n",
    "        \"uniprot_end\": None,\n",
    "        \"latest_version\": None,\n",
    "    }\n",
    "\n",
    "    if meta_status not in (\"success\", \"cached\"):\n",
    "        return record\n",
    "\n",
    "    entry = metadata\n",
    "    record[\"uniprot_sequence\"] = entry.get(\"uniprotSequence\")\n",
    "    record[\"uniprot_start\"] = entry.get(\"uniprotStart\")\n",
    "    record[\"uniprot_end\"] = entry.get(\"uniprotEnd\")\n",
    "    record[\"latest_version\"] = entry.get(\"latestVersion\")\n",
    "\n",
    "    pdb_path = os.path.join(PDB_DIR, f\"{uniprot_id}.pdb\")\n",
    "    pae_path = os.path.join(PAE_DIR, f\"{uniprot_id}.json\")\n",
    "\n",
    "    record[\"pdb_status\"] = download_one(session, entry[\"pdbUrl\"], pdb_path)\n",
    "    record[\"pae_status\"] = download_one(session, entry[\"paeDocUrl\"], pae_path)\n",
    "\n",
    "    if record[\"pdb_status\"] in (\"success\", \"already_exists\"):\n",
    "        record[\"pdb_path\"] = pdb_path\n",
    "    if record[\"pae_status\"] in (\"success\", \"already_exists\"):\n",
    "        record[\"pae_path\"] = pae_path\n",
    "\n",
    "    return record\n",
    "\n",
    "\n",
    "MAX_WORKERS = 8\n",
    "results = []\n",
    "\n",
    "print(f\"Fetching metadata + assets for {len(unique_uniprots):,} proteins ({MAX_WORKERS} parallel workers)...\")\n",
    "with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:\n",
    "    futures = {executor.submit(process_protein, uid): uid for uid in unique_uniprots}\n",
    "    for future in tqdm(as_completed(futures), total=len(futures)):\n",
    "        results.append(future.result())\n",
    "\n",
    "print(f\"\\nPipeline pass complete. {len(results):,} proteins processed.\")"
))

cells.append(md(
    "## 6. Structural Sequence Re-Validation\n",
    "This is the key correctness check requested for this phase: we confirm that `(Ref_AA, Pos)` for every variant matches "
    "the residue at that position in `uniprot_sequence` — the sequence AlphaFold metadata states it actually modelled — "
    "**independently of** Phase 1's check against the live UniProt FASTA endpoint. A mismatch here, that did *not* show "
    "up in Phase 1, is direct evidence of sequence drift between the UniProt snapshot AlphaFold was built from and the "
    "UniProt snapshot Phase 1 queried. Any such variant is excluded — training the GNN on a residue that AlphaFold did "
    "not actually model at that position would silently corrupt the structural signal."
))

cells.append(code(
    "manifest_df = pd.DataFrame(results)\n",
    "\n",
    "manifest_df[\"meta_ok\"] = manifest_df[\"meta_status\"].isin([\"success\", \"cached\"])\n",
    "manifest_df[\"pdb_ok\"] = manifest_df[\"pdb_status\"].isin([\"success\", \"already_exists\"])\n",
    "manifest_df[\"pae_ok\"] = manifest_df[\"pae_status\"].isin([\"success\", \"already_exists\"])\n",
    "manifest_df[\"assets_ok\"] = manifest_df[\"meta_ok\"] & manifest_df[\"pdb_ok\"] & manifest_df[\"pae_ok\"]\n",
    "\n",
    "aa_map = {\n",
    "    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',\n",
    "    'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'His': 'H', 'Ile': 'I',\n",
    "    'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',\n",
    "    'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V'\n",
    "}\n",
    "\n",
    "seq_by_uniprot = manifest_df.set_index(\"UniProt_ID\")[\"uniprot_sequence\"].to_dict()\n",
    "\n",
    "def revalidate_against_alphafold(row):\n",
    "    seq = seq_by_uniprot.get(row[\"UniProt_ID\"])\n",
    "    if not seq:\n",
    "        return False, \"No AlphaFold Sequence (assets unavailable)\"\n",
    "    pos = int(row[\"Pos\"])\n",
    "    if pos < 1 or pos > len(seq):\n",
    "        return False, \"Position Out of Bounds (AlphaFold sequence)\"\n",
    "    expected = aa_map.get(row[\"Ref_AA\"])\n",
    "    actual = seq[pos - 1]\n",
    "    if expected == actual:\n",
    "        return True, \"Structurally Valid\"\n",
    "    return False, f\"Sequence Drift: Phase 1 expected {expected}, AlphaFold model has {actual}\"\n",
    "\n",
    "print(\"Re-validating Ref_AA/Pos against the exact sequence AlphaFold modelled...\")\n",
    "tqdm.pandas()\n",
    "struct_results = df.progress_apply(revalidate_against_alphafold, axis=1)\n",
    "df[\"Structurally_Valid\"] = [r[0] for r in struct_results]\n",
    "df[\"Structural_Validation_Reason\"] = [r[1] for r in struct_results]\n",
    "\n",
    "print(\"\\n=== Structural re-validation summary ===\")\n",
    "print(df[\"Structural_Validation_Reason\"].value_counts().to_string())"
))

cells.append(md(
    "## 7. Summary & Manifest\n",
    "The manifest records, per protein, every stage's outcome (metadata, PDB, PAE) so failures can be traced to their "
    "cause. This is the artifact Phase 3 should consult to know which proteins are safe to build graphs from."
))

cells.append(code(
    "print(\"=== Metadata fetch status ===\")\n",
    "print(manifest_df[\"meta_status\"].value_counts().to_string())\n",
    "print(\"\\n=== PDB download status ===\")\n",
    "print(manifest_df[\"pdb_status\"].value_counts().to_string())\n",
    "print(\"\\n=== PAE download status ===\")\n",
    "print(manifest_df[\"pae_status\"].value_counts().to_string())\n",
    "print(f\"\\nProteins with complete, verified assets : {manifest_df['assets_ok'].sum():,} / {len(manifest_df):,}\")\n",
    "\n",
    "manifest_file = \"asset_download_manifest.csv\"\n",
    "manifest_df.drop(columns=[\"uniprot_sequence\"]).to_csv(manifest_file, index=False)\n",
    "print(f\"\\nManifest saved to {manifest_file} (sequence column omitted from CSV for size; kept in memory for Section 6)\")\n",
    "\n",
    "failed_df = manifest_df[~manifest_df[\"assets_ok\"]]\n",
    "if len(failed_df) > 0:\n",
    "    print(f\"\\n{len(failed_df):,} proteins missing at least one verified asset:\")\n",
    "    print(failed_df[[\"UniProt_ID\", \"meta_status\", \"pdb_status\", \"pae_status\"]].head(20).to_string(index=False))"
))

cells.append(md(
    "## 8. Filter Phase 1 Variants to Fully Verified Proteins\n",
    "A variant is carried forward to Phase 3 only if **all** of the following hold:\n"
    "1. Its protein's AlphaFold metadata, PDB, and PAE were all successfully obtained (`assets_ok`).\n"
    "2. The variant passed the original Phase 1 validation (`Ref_AA` matched UniProt at the time of Phase 1).\n"
    "3. The variant passed this phase's independent structural re-validation (`Ref_AA` matches the sequence AlphaFold "
    "actually modelled — `Structurally_Valid`).\n\n"
    "Requiring both (2) and (3) — rather than only the more recent check — ensures the variant is correct under *both* "
    "UniProt snapshots, which is the more conservative and defensible standard for a structure-conditioned model."
))

cells.append(code(
    "verified_proteins = set(manifest_df.loc[manifest_df[\"assets_ok\"], \"UniProt_ID\"])\n",
    "\n",
    "before = len(df)\n",
    "df_with_assets = df[\n",
    "    df[\"UniProt_ID\"].isin(verified_proteins) & df[\"Structurally_Valid\"]\n",
    "].copy()\n",
    "\n",
    "print(f\"Variants before Phase 2 filter : {before:,}\")\n",
    "print(f\"Variants after Phase 2 filter  : {len(df_with_assets):,} (dropped {before - len(df_with_assets):,})\")\n",
    "print(f\"  Pathogenic (1): {(df_with_assets['Target'] == 1).sum():,}\")\n",
    "print(f\"  Benign     (0): {(df_with_assets['Target'] == 0).sum():,}\")\n",
    "print(f\"Unique proteins carried to Phase 3 : {df_with_assets['UniProt_ID'].nunique():,}\")\n",
    "\n",
    "output_file = \"clinvar_with_structures.csv\"\n",
    "df_with_assets.to_csv(output_file, index=False)\n",
    "print(f\"\\nSaved to {output_file} — input for Phase 3 (graph construction)\")"
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

out_path = r"c:\Users\Megha Kanojia\Documents\DevProjects\agy2-projects\missense-variants\phase2_asset_download.ipynb"
with open(out_path, "w") as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook written to {out_path}")
