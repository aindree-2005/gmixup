import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN

from torch_geometric.nn import (
    GCNConv,
    GINConv,
    GraphConv,
    TopKPooling,
    global_mean_pool,
    global_mean_pool as gap,
    global_max_pool as gmp,
    DenseGraphConv,
    dense_diff_pool,
    dense_mincut_pool,
    TransformerConv,
)

from torch_geometric.utils import to_dense_batch, to_dense_adj

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, DenseGraphConv, dense_mincut_pool, dense_diff_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.nn import TransformerConv
class GIN(torch.nn.Module):
    def __init__(self, num_features=1, num_classes=1, num_hidden=32):
        super(GIN, self).__init__()

        # if data.x is None:
        #   data.x = torch.ones([data.num_nodes, 1], dtype=torch.float)
        # dataset.data.edge_attr = None

        # num_features = dataset.num_features
        dim = num_hidden

        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        # x = global_add_pool(x, batch)
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)
class GCN(torch.nn.Module):
    def __init__(self, num_features=1, num_classes=1, num_hidden=64):
        super(GCN, self).__init__()

        dim = num_hidden

        self.conv1 = GCNConv(num_features, dim)
        self.bn1 = BN(dim)

        self.conv2 = GCNConv(dim, dim)
        self.bn2 = BN(dim)

        self.conv3 = GCNConv(dim, dim)
        self.bn3 = BN(dim)

        self.conv4 = GCNConv(dim, dim)
        self.bn4 = BN(dim)

        self.fc = Linear(dim, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)

        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)

        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)

        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)

        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)
class TopKPool(torch.nn.Module):
    def __init__(self, num_features=1, num_classes=1, num_hidden=64, ratio=0.8):
        super(TopKPool, self).__init__()

        dim = num_hidden

        self.conv1 = GraphConv(num_features, dim)
        self.pool1 = TopKPooling(dim, ratio)

        self.conv2 = GraphConv(dim, dim)
        self.pool2 = TopKPooling(dim, ratio)

        self.conv3 = GraphConv(dim, dim)
        self.pool3 = TopKPooling(dim, ratio)

        self.fc1 = Linear(dim * 2, dim)
        self.fc2 = Linear(dim, dim)
        self.fc3 = Linear(dim, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)

        x = torch.cat([gap(x, batch), gmp(x, batch)], dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.log_softmax(x, dim=-1)
class GMT(torch.nn.Module):
    def __init__(self, num_features=1, num_classes=1, num_hidden=64, heads=4):
        super(GMT, self).__init__()

        dim = num_hidden

        self.conv1 = TransformerConv(num_features, dim, heads=heads, concat=True)
        self.conv2 = TransformerConv(dim * heads, dim, heads=heads, concat=True)

        self.fc1 = Linear(dim * heads, dim)
        self.fc2 = Linear(dim, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        x = global_mean_pool(x, batch)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=-1)

class DiffPool(torch.nn.Module):
    def __init__(self, num_features=1, num_classes=1, num_hidden=64, max_nodes=100):
        super(DiffPool, self).__init__()
        dim = num_hidden
        self.max_nodes = max_nodes
        # GNN for node embeddings
        self.gnn_embed = DenseGraphConv(num_features, dim)
        # GNN for assignment matrix
        self.gnn_pool = DenseGraphConv(num_features, max_nodes)
        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, num_classes)

    def forward(self, x, edge_index, batch):
        # Convert to dense
        x, mask = to_dense_batch(x, batch)
        adj = to_dense_adj(edge_index, batch)
        # Compute embeddings and assignment
        z = F.relu(self.gnn_embed(x, adj))
        s = F.softmax(self.gnn_pool(x, adj), dim=-1)
        # DiffPool
        x, adj, link_loss, ent_loss = dense_diff_pool(z, adj, s, mask)
        # Readout
        x = x.mean(dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)
class MinCutPool(torch.nn.Module):
    def __init__(self, num_features=1, num_classes=1, num_hidden=64, max_nodes=100):
        super(MinCutPool, self).__init__()

        dim = num_hidden
        self.max_nodes = max_nodes

        self.gnn_embed = DenseGraphConv(num_features, dim)
        self.gnn_pool = DenseGraphConv(num_features, max_nodes)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, num_classes)

    def forward(self, x, edge_index, batch):
        x, mask = to_dense_batch(x, batch)
        adj = to_dense_adj(edge_index, batch)

        z = F.relu(self.gnn_embed(x, adj))
        s = F.softmax(self.gnn_pool(x, adj), dim=-1)

        x, adj, mincut_loss, ortho_loss = dense_mincut_pool(
            z, adj, s, mask
        )

        x = x.mean(dim=1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=-1)
