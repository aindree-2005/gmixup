import os
import csv
import random
import argparse
import numpy as np
from collections import Counter

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree

from sklearn.metrics import f1_score

from utils import (
    stat_graph,
    split_class_graphs,
    align_graphs,
    two_graphons_mixup,
    universal_svd,
)

from models import GIN, GCN, TopKPool

import os
import csv
import random
import argparse
import numpy as np
from collections import Counter

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree

from sklearn.metrics import f1_score

from utils import (
    stat_graph,
    split_class_graphs,
    align_graphs,
    two_graphons_mixup,
    universal_svd,
)

from models import GIN, GCN, TopKPool


def prepare_dataset_x(dataset):
    max_degree = 0
    degs = []

    for g in dataset:
        d = degree(g.edge_index[0], dtype=torch.long)
        degs.append(d)
        max_degree = max(max_degree, d.max().item())
        g.num_nodes = int(torch.max(g.edge_index)) + 1

    if max_degree < 2000:
        for g in dataset:
            d = degree(g.edge_index[0], dtype=torch.long)
            g.x = F.one_hot(d, num_classes=max_degree + 1).float()
    else:
        all_deg = torch.cat(degs).float()
        mean, std = all_deg.mean(), all_deg.std()
        for g in dataset:
            d = degree(g.edge_index[0], dtype=torch.float)
            g.x = ((d - mean) / std).view(-1, 1)

    return dataset


def prepare_dataset_onehot_y(dataset):
    classes = sorted(set(int(g.y) for g in dataset))
    C = len(classes)
    for g in dataset:
        g.y = F.one_hot(g.y, num_classes=C).float()[0]
    return dataset


def mixup_loss(pred, target):
    return -(pred * target).sum(dim=1).mean()


def split_proteins_imbalanced(dataset):
    labels = [int(g.y.argmax().item()) for g in dataset]

    c0 = [g for g, y in zip(dataset, labels) if y == 0]
    c1 = [g for g, y in zip(dataset, labels) if y == 1]

    random.shuffle(c0)
    random.shuffle(c1)

    train = c0[:30] + c1[:270]
    test = c0[30:] + c1[270:]

    return train, test


def train_epoch(model, loader, opt, device, C):
    model.train()
    for d in loader:
        d = d.to(device)
        opt.zero_grad()
        out = model(d)
        loss = mixup_loss(out, d.y.view(-1, C))
        loss.backward()
        opt.step()


@torch.no_grad()
def eval_model(model, loader, device, C):
    model.eval()
    yt, yp = [], []

    for d in loader:
        d = d.to(device)
        out = model(d)
        yp.extend(out.argmax(1).cpu().numpy())
        yt.extend(d.y.view(-1, C).argmax(1).cpu().numpy())

    yt = np.array(yt)
    yp = np.array(yp)

    acc = np.mean(yt == yp)
    f1_macro = f1_score(yt, yp, average="macro", zero_division=0)
    f1_micro = f1_score(yt, yp, average="micro", zero_division=0)

    return acc, f1_macro, f1_micro


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = list(TUDataset("data", "PROTEINS"))
    for g in dataset:
        g.y = g.y.view(-1)

    dataset = prepare_dataset_onehot_y(dataset)
    random.shuffle(dataset)

    train_set, test_set = split_proteins_imbalanced(dataset)
    print("Initial train dist:", Counter(int(g.y.argmax()) for g in train_set))

    class_graphs = split_class_graphs(train_set)
    _, _, _, med_nodes, _, _ = stat_graph(train_set)
    N = int(med_nodes)

    graphons = []
    for label, graphs in class_graphs:
        aligned, _, _, _ = align_graphs(graphs, padding=True, N=N)
        graphon = universal_svd(aligned, threshold=0.2)
        graphons.append((label, graphon))

    minority = min(
        Counter(int(g.y.argmax()) for g in train_set),
        key=lambda k: Counter(int(g.y.argmax()) for g in train_set)[k]
    )

    target = max(Counter(int(g.y.argmax()) for g in train_set).values())
    need = target - Counter(int(g.y.argmax()) for g in train_set)[minority]

    minority_graphon = [g for g in graphons if int(np.argmax(g[0])) == minority][0]

    new_graphs = []
    lam_list = np.random.uniform(*args.lam_range, args.aug_num)
    for lam in lam_list:
        new_graphs += two_graphons_mixup(
            [minority_graphon, minority_graphon],
            la=lam,
            num_sample=int(np.ceil(need / args.aug_num))
        )

    print("Synthetic dist:", Counter(int(g.y.argmax()) for g in new_graphs))

    full_dataset = train_set + new_graphs + test_set
    full_dataset = prepare_dataset_x(full_dataset)

    train_set = full_dataset[:len(train_set) + len(new_graphs)]
    test_set = full_dataset[len(train_set):]

    print("Final train dist:", Counter(int(g.y.argmax()) for g in train_set))

    Fdim = train_set[0].x.size(1)
    C = train_set[0].y.size(0)

    loaders = {
        "train": DataLoader(train_set, args.batch_size, shuffle=True),
        "test": DataLoader(test_set, args.batch_size),
    }

    models = {
        "GIN": GIN,
        "GCN": GCN,
        "TopKPool": TopKPool,
    }

    for name, cls in models.items():
        model = cls(Fdim, C, args.hidden_dim).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

        for _ in range(args.epochs):
            train_epoch(model, loaders["train"], opt, device, C)

        acc, f1_macro, f1_micro = eval_model(model, loaders["test"], device, C)
        print(
            f"{name} | "
            f"Acc={acc:.4f} | "
            f"F1-macro={f1_macro:.4f} | "
            f"F1-micro={f1_micro:.4f}"
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--aug_num", type=int, default=10)
    p.add_argument("--lam_range", type=float, nargs=2, default=[0.005, 0.01])
    args = p.parse_args()

    run(args)

def prepare_dataset_x(dataset):
    max_degree = 0
    degs = []

    for g in dataset:
        d = degree(g.edge_index[0], dtype=torch.long)
        degs.append(d)
        max_degree = max(max_degree, d.max().item())
        g.num_nodes = int(torch.max(g.edge_index)) + 1

    if max_degree < 2000:
        for g in dataset:
            d = degree(g.edge_index[0], dtype=torch.long)
            g.x = F.one_hot(d, num_classes=max_degree + 1).float()
    else:
        all_deg = torch.cat(degs).float()
        mean, std = all_deg.mean(), all_deg.std()
        for g in dataset:
            d = degree(g.edge_index[0], dtype=torch.float)
            g.x = ((d - mean) / std).view(-1, 1)

    return dataset


def prepare_dataset_onehot_y(dataset):
    classes = sorted(set(int(g.y) for g in dataset))
    C = len(classes)
    for g in dataset:
        g.y = F.one_hot(g.y, num_classes=C).float()[0]
    return dataset


def mixup_loss(pred, target):
    return -(pred * target).sum(dim=1).mean()


def split_proteins_imbalanced(dataset):
    labels = [int(g.y.argmax().item()) for g in dataset]

    c0 = [g for g, y in zip(dataset, labels) if y == 0]
    c1 = [g for g, y in zip(dataset, labels) if y == 1]

    random.shuffle(c0)
    random.shuffle(c1)

    train = c0[:30] + c1[:270]
    test = c0[30:] + c1[270:]

    return train, test


def train_epoch(model, loader, opt, device, C):
    model.train()
    for d in loader:
        d = d.to(device)
        opt.zero_grad()
        out = model(d)
        loss = mixup_loss(out, d.y.view(-1, C))
        loss.backward()
        opt.step()


@torch.no_grad()
def eval_model(model, loader, device, C):
    model.eval()
    yt, yp = [], []

    for d in loader:
        d = d.to(device)
        out = model(d)
        yp.extend(out.argmax(1).cpu().numpy())
        yt.extend(d.y.view(-1, C).argmax(1).cpu().numpy())

    yt = np.array(yt)
    yp = np.array(yp)

    acc = np.mean(yt == yp)
    f1_macro = f1_score(yt, yp, average="macro", zero_division=0)
    f1_micro = f1_score(yt, yp, average="micro", zero_division=0)

    return acc, f1_macro, f1_micro


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = list(TUDataset("data", "PROTEINS"))
    for g in dataset:
        g.y = g.y.view(-1)

    dataset = prepare_dataset_onehot_y(dataset)
    random.shuffle(dataset)

    train_set, test_set = split_proteins_imbalanced(dataset)
    print("Initial train dist:", Counter(int(g.y.argmax()) for g in train_set))

    class_graphs = split_class_graphs(train_set)
    _, _, _, med_nodes, _, _ = stat_graph(train_set)
    N = int(med_nodes)

    graphons = []
    for label, graphs in class_graphs:
        aligned, _, _, _ = align_graphs(graphs, padding=True, N=N)
        graphon = universal_svd(aligned, threshold=0.2)
        graphons.append((label, graphon))

    minority = min(
        Counter(int(g.y.argmax()) for g in train_set),
        key=lambda k: Counter(int(g.y.argmax()) for g in train_set)[k]
    )

    target = max(Counter(int(g.y.argmax()) for g in train_set).values())
    need = target - Counter(int(g.y.argmax()) for g in train_set)[minority]

    minority_graphon = [g for g in graphons if int(np.argmax(g[0])) == minority][0]

    new_graphs = []
    lam_list = np.random.uniform(*args.lam_range, args.aug_num)
    for lam in lam_list:
        new_graphs += two_graphons_mixup(
            [minority_graphon, minority_graphon],
            la=lam,
            num_sample=int(np.ceil(need / args.aug_num))
        )

    print("Synthetic dist:", Counter(int(g.y.argmax()) for g in new_graphs))

    full_dataset = train_set + new_graphs + test_set
    full_dataset = prepare_dataset_x(full_dataset)

    train_set = full_dataset[:len(train_set) + len(new_graphs)]
    test_set = full_dataset[len(train_set):]

    print("Final train dist:", Counter(int(g.y.argmax()) for g in train_set))

    Fdim = train_set[0].x.size(1)
    C = train_set[0].y.size(0)

    loaders = {
        "train": DataLoader(train_set, args.batch_size, shuffle=True),
        "test": DataLoader(test_set, args.batch_size),
    }

    models = {
        "GIN": GIN,
        "GCN": GCN,
        "TopKPool": TopKPool
    }

    for name, cls in models.items():
        model = cls(Fdim, C, args.hidden_dim).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

        for _ in range(args.epochs):
            train_epoch(model, loaders["train"], opt, device, C)

        acc, f1_macro, f1_micro = eval_model(model, loaders["test"], device, C)
        print(
            f"{name} | "
            f"Acc={acc:.4f} | "
            f"F1-macro={f1_macro:.4f} | "
            f"F1-micro={f1_micro:.4f}"
        )
        


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--aug_num", type=int, default=10)
    p.add_argument("--lam_range", type=float, nargs=2, default=[0.005, 0.01])
    args = p.parse_args()

    run(args)
