"""
Shared model architectures for the missense-variant GNN project.

Kept in a module (not pasted inline in a notebook) specifically so a saved checkpoint's
state_dict can be reloaded later - e.g. from Phase 5 (evaluation/benchmarking) - by importing
the exact same class definition used during training, rather than redefining it by hand and
risking an architecture mismatch that silently loads incorrectly or errors out.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv


class GATv2Classifier(nn.Module):
    """GATv2-based variant classifier. Set use_edge_attr=False for the unweighted ablation."""

    def __init__(self, in_dim, hidden_dim=128, heads=4, edge_dim=2, use_edge_attr=True, dropout=0.2):
        super().__init__()
        self.use_edge_attr = use_edge_attr
        self.proj = nn.Linear(in_dim, hidden_dim)
        ed = edge_dim if use_edge_attr else None
        self.conv1 = GATv2Conv(hidden_dim, hidden_dim, heads=heads, edge_dim=ed, concat=False, dropout=dropout)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=heads, edge_dim=ed, concat=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, batch, return_attention=False):
        x = torch.relu(self.proj(batch.x))
        ea = batch.edge_attr if self.use_edge_attr else None
        if return_attention:
            x, (edge_index_1, alpha_1) = self.conv1(x, batch.edge_index, ea, return_attention_weights=True)
            x = torch.relu(x)
            x = self.dropout(x)
            x, (edge_index_2, alpha_2) = self.conv2(x, batch.edge_index, ea, return_attention_weights=True)
            x = torch.relu(x)
            global_idx = batch.ptr[:-1] + batch.variant_node_idx
            logits = self.classifier(x[global_idx]).squeeze(-1)
            return logits, (edge_index_1, alpha_1, edge_index_2, alpha_2)

        x = torch.relu(self.conv1(x, batch.edge_index, ea))
        x = self.dropout(x)
        x = torch.relu(self.conv2(x, batch.edge_index, ea))
        global_idx = batch.ptr[:-1] + batch.variant_node_idx
        return self.classifier(x[global_idx]).squeeze(-1)


class GCNClassifier(nn.Module):
    """Plain GCN baseline - no attention, no edge features."""

    def __init__(self, in_dim, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, batch):
        x = torch.relu(self.proj(batch.x))
        x = torch.relu(self.conv1(x, batch.edge_index))
        x = self.dropout(x)
        x = torch.relu(self.conv2(x, batch.edge_index))
        global_idx = batch.ptr[:-1] + batch.variant_node_idx
        return self.classifier(x[global_idx]).squeeze(-1)


class GraphSAGEClassifier(nn.Module):
    """GraphSAGE baseline - neighbor-sampling-style mean aggregation, no attention, no edge features."""

    def __init__(self, in_dim, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.conv1 = SAGEConv(hidden_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, batch):
        x = torch.relu(self.proj(batch.x))
        x = torch.relu(self.conv1(x, batch.edge_index))
        x = self.dropout(x)
        x = torch.relu(self.conv2(x, batch.edge_index))
        global_idx = batch.ptr[:-1] + batch.variant_node_idx
        return self.classifier(x[global_idx]).squeeze(-1)
