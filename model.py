import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class GraphConvLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, out_dim):
        super().__init__()
        self.node_transform = nn.Linear(node_dim, hidden_dim)
        self.edge_transform = nn.Linear(edge_dim, hidden_dim)
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim))
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim))

    def forward(self, node_features, edge_indices, edge_features):
        node_emb = self.node_transform(node_features)
        edge_emb = self.edge_transform(edge_features)
        src, dst = edge_indices
        messages = torch.cat([node_emb[src], edge_emb], dim=-1)
        messages = self.message_mlp(messages)
        aggregated = torch.zeros_like(node_emb[:, :messages.size(-1)])
        aggregated.index_add_(0, dst, messages)
        updated = torch.cat([node_features, aggregated], dim=-1)
        return self.update_mlp(updated)

class TacticalGNN(nn.Module):
    def __init__(self, node_dim=14, edge_dim=4, hidden_dim=128, num_layers=4):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.conv_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        current_dim = node_dim
        for _ in range(num_layers):
            out_dim = hidden_dim
            self.conv_layers.append(GraphConvLayer(current_dim, edge_dim, hidden_dim, out_dim))
            self.layer_norms.append(nn.LayerNorm(out_dim))
            current_dim = out_dim
        self.move_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4))
        self.shooter_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1))
        self.target_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1))

    def _compute_target(self, x):
        scores = []
        for i in range(x.size(0)):
            for j in range(x.size(0)):
                if i == j:
                    continue
                combined = torch.cat([x[i], x[j]], dim=-1)
                scores.append(self.target_head(combined).squeeze(-1))
        if scores:
            return torch.stack(scores)
        return torch.tensor([], device=x.device)

    def forward(self, graph_data):
        node_features = graph_data['node_features']
        edge_indices = graph_data['edge_indices']
        edge_features = graph_data['edge_features']
        x = node_features
        for conv, norm in zip(self.conv_layers, self.layer_norms):
            residual = x if x.size(-1) == self.hidden_dim else None
            x = conv(x, edge_indices, edge_features)
            x = norm(x)
            if residual is not None:
                x = x + residual
            x = F.relu(x)
        move_logits = self.move_head(x)                  # [N,4]
        move_probs = F.softmax(move_logits, dim=-1)
        shooter_logits = self.shooter_head(x).squeeze(-1)
        target_scores = self._compute_target(x)
        return {
            'move_logits': move_logits,
            'move_probs': torch.clamp(move_probs, 1e-3, 1-1e-3),
            'shooter_logits': torch.clamp(shooter_logits, -10, 10),
            'target_scores': torch.clamp(target_scores, -10, 10),
            'node_embeddings': x}

