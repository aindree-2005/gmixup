import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN

from torch_geometric.nn import (
    GCNConv,
    GINConv,
    GraphConv,
    TopKPooling,
    TransformerConv,
    global_mean_pool,
    global_max_pool,
)

class GIN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=64):
        super().__init__()

        def mlp(in_dim, out_dim):
            return Sequential(
                Linear(in_dim, out_dim),
                ReLU(),
                Linear(out_dim, out_dim)
            )

        self.conv1 = GINConv(mlp(num_features, hidden_dim))
        self.bn1 = BN(hidden_dim)

        self.conv2 = GINConv(mlp(hidden_dim, hidden_dim))
        self.bn2 = BN(hidden_dim)

        self.conv3 = GINConv(mlp(hidden_dim, hidden_dim))
        self.bn3 = BN(hidden_dim)

        self.conv4 = GINConv(mlp(hidden_dim, hidden_dim))
        self.bn4 = BN(hidden_dim)

        self.conv5 = GINConv(mlp(hidden_dim, hidden_dim))
        self.bn5 = BN(hidden_dim)

        self.fc1 = Linear(hidden_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x = F.relu(self.bn4(self.conv4(x, edge_index)))
        x = F.relu(self.bn5(self.conv5(x, edge_index)))

        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=-1)

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=64):
        super().__init__()

        self.conv1 = GCNConv(num_features, hidden_dim)
        self.bn1 = BN(hidden_dim)

        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = BN(hidden_dim)

        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = BN(hidden_dim)

        self.conv4 = GCNConv(hidden_dim, hidden_dim)
        self.bn4 = BN(hidden_dim)

        self.fc = Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x = F.relu(self.bn4(self.conv4(x, edge_index)))

        x = global_mean_pool(x, batch)
        x = self.fc(x)

        return F.log_softmax(x, dim=-1)
    
class TopKPool(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=64, ratio=0.8):
        super().__init__()

        self.conv1 = GraphConv(num_features, hidden_dim)
        self.pool1 = TopKPooling(hidden_dim, ratio)

        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.pool2 = TopKPooling(hidden_dim, ratio)

        self.conv3 = GraphConv(hidden_dim, hidden_dim)
        self.pool3 = TopKPooling(hidden_dim, ratio)

        self.fc1 = Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = Linear(hidden_dim, hidden_dim)
        self.fc3 = Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)

        x = torch.cat(
            [global_mean_pool(x, batch), global_max_pool(x, batch)],
            dim=1
        )

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.log_softmax(x, dim=-1)

class GMT(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=16, heads=2):
        super().__init__()

        self.conv1 = TransformerConv(
            num_features, hidden_dim, heads=heads, concat=True
        )
        self.conv2 = TransformerConv(
            hidden_dim * heads, hidden_dim, heads=heads, concat=True
        )

        self.fc1 = Linear(hidden_dim * heads, hidden_dim)
        self.fc2 = Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=-1)
from torch_geometric.nn import GCN2Conv, global_mean_pool
class GCN2(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        hidden_dim=64,
        num_layers=8,
        alpha=0.1,
        theta=0.5,
        dropout=0.5
    ):
        super().__init__()

        self.dropout = dropout
        self.num_layers = num_layers
        self.lin1 = Linear(num_features, hidden_dim)
        self.lin2 = Linear(hidden_dim, num_classes)
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(
                    hidden_dim,
                    alpha=alpha,
                    theta=theta,
                    layer=layer + 1
                )
            )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))

        x0 = x   
        for conv in self.convs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(conv(x, x0, edge_index))
        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1)
