"""
Shared dataset code for the missense-variant GNN project.

Extracted from phase3_graph_construction.ipynb (Section 6) so that Phase 4 (and any later
phase) can import the exact same MissenseVariantGraphDataset rather than re-pasting it and
risking the two copies drifting apart.
"""
import os
from collections import OrderedDict

import torch
from torch_geometric.data import Data, Dataset

ESM2_EMBEDDING_DIM = 1280

AA_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
    'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
    'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}
AA_1TO_ONEHOT_IDX = {aa: i for i, aa in enumerate(sorted(AA_3TO1.values()))}
N_AA = len(AA_1TO_ONEHOT_IDX)

VARIANT_FLAG_COL = ESM2_EMBEDDING_DIM + N_AA          # column 1300
ALT_AA_START_COL = VARIANT_FLAG_COL + 1                # columns 1301..1320
NODE_FEATURE_DIM = ESM2_EMBEDDING_DIM + N_AA + 1 + N_AA  # 1321


class MissenseVariantGraphDataset(Dataset):
    """
    Each item is one ClinVar missense variant, represented as the shared structural graph for its
    protein with two node-level edits applied at the variant's position: a variant-site flag and the
    substituted (Alt_AA) one-hot identity. Label (y) is the ClinVar pathogenicity Target
    (1 = pathogenic, 0 = benign). `variant_node_idx` is the LOCAL node index of the mutated residue
    within this graph - used by the model to read out that node's embedding after message passing.
    """

    def __init__(self, variant_df, graph_dir, cache_size=64):
        super().__init__()
        self.variant_df = variant_df.reset_index(drop=True)
        self.graph_dir = graph_dir
        self.cache_size = cache_size
        self._graph_cache = OrderedDict()

    def len(self):
        return len(self.variant_df)

    def _load_base_graph(self, uniprot_id):
        if uniprot_id in self._graph_cache:
            self._graph_cache.move_to_end(uniprot_id)
            return self._graph_cache[uniprot_id]
        graph = torch.load(os.path.join(self.graph_dir, f"{uniprot_id}.pt"), weights_only=False)
        self._graph_cache[uniprot_id] = graph
        if len(self._graph_cache) > self.cache_size:
            self._graph_cache.popitem(last=False)
        return graph

    def get(self, idx):
        row = self.variant_df.iloc[idx]
        base_graph = self._load_base_graph(row["UniProt_ID"])

        x = base_graph.x.clone()
        node_idx = int(row["Pos"]) - 1

        x[node_idx, VARIANT_FLAG_COL] = 1.0
        alt_idx = AA_1TO_ONEHOT_IDX[row["Alt_AA_1"]]
        x[node_idx, ALT_AA_START_COL + alt_idx] = 1.0

        return Data(
            x=x,
            edge_index=base_graph.edge_index,
            edge_attr=base_graph.edge_attr,
            y=torch.tensor([row["Target"]], dtype=torch.float),
            variant_node_idx=torch.tensor([node_idx], dtype=torch.long),
            num_nodes=x.shape[0],
            uniprot_id=row["UniProt_ID"],
            variation_id=row["VariationID"],
        )
