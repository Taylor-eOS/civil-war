import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class GraphConvLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.node_lin = nn.Linear(node_dim, hidden_dim)
        self.edge_lin = nn.Linear(edge_dim, hidden_dim)
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))

    def forward(self, node_feats, edge_idx, edge_feats):
        h = self.node_lin(node_feats)
        e = self.edge_lin(edge_feats)
        src, dst = edge_idx
        m = self.msg_mlp(torch.cat([h[src], e], dim=-1))
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, m)
        up = torch.cat([node_feats, agg], dim=-1)
        return self.update_mlp(up)

class TacticalGNN(nn.Module):
    def __init__(self, node_dim, edge_dim):
        super().__init__()
        self.hidden_dim = 256
        self.num_layers = 6
        self.conv_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        dims = [node_dim] + [self.hidden_dim] * self.num_layers
        for i in range(self.num_layers):
            self.conv_layers.append(GraphConvLayer(dims[i], edge_dim, self.hidden_dim))
            self.layer_norms.append(nn.LayerNorm(self.hidden_dim))
        self.global_lin = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.move_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, 4))
        self.shooter_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, 1))
        self.target_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, 1))

    def forward(self, graph_data):
        x = graph_data['node_features']
        ei = graph_data['edge_indices']
        ef = graph_data['edge_features']
        for conv, norm in zip(self.conv_layers, self.layer_norms):
            res = x
            x = conv(x, ei, ef)
            x = norm(x)
            if res.size(-1) == x.size(-1):
                x = F.relu(x + res)
            else:
                x = F.relu(x)
        global_ctx = self.global_lin(x.mean(0, keepdim=True)).expand(x.size(0), -1)
        mv = self.move_head(torch.cat([x, global_ctx], dim=-1))
        sv = self.shooter_head(torch.cat([x, global_ctx], dim=-1)).squeeze(-1)
        ts = []
        n = x.size(0)
        for i in range(n):
            for j in range(n):
                if i != j:
                    ts.append(self.target_head(torch.cat([x[i], x[j], global_ctx[i]], dim=-1)).squeeze(-1))
        ts = torch.stack(ts) if ts else torch.tensor([], device=x.device)
        return {
            'move_logits': mv,
            'move_probs': F.softmax(mv, dim=-1),
            'shooter_logits': torch.clamp(sv, -10, 10),
            'target_scores': torch.clamp(ts, -10, 10),
            'node_embeddings': x}

