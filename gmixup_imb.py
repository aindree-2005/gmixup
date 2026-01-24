import os
import csv
import random
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree

from utils import (
    stat_graph,
    split_class_graphs,
    align_graphs,
    two_graphons_mixup,
    universal_svd,
)

from models import GIN, GCN, TopKPool, GCN2

def drop_edge_attributes(dataset):
    for g in dataset:
        if hasattr(g, "edge_attr"):
            del g.edge_attr
    return dataset

def prepare_dataset_x(dataset):
    if dataset[0].x is None:
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
    y_set = set(int(g.y) for g in dataset)
    num_classes = len(y_set)
    for g in dataset:
        g.y = F.one_hot(g.y, num_classes=num_classes).float()[0]
    return dataset


def get_class_stats(dataset):
    ys = torch.stack([g.y for g in dataset])
    counts = ys.sum(dim=0)
    freqs = counts / counts.sum()
    return counts, freqs


def onehot_to_index(label):
    if isinstance(label, (list, np.ndarray)) and len(label) > 1:
        return int(np.argmax(label))
    if torch.is_tensor(label) and label.numel() > 1:
        return int(torch.argmax(label).item())
    return int(label)

def weighted_mixup_loss(pred, target, class_weights):
    weights = (target * class_weights).sum(dim=1)
    loss = -(pred * target).sum(dim=1)
    return (weights * loss).mean()

def train_epoch(model, loader, optimizer, device, num_classes, class_weights):
    model.train()
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        y = data.y.view(-1, num_classes)
        loss = weighted_mixup_loss(out, y, class_weights)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def eval_model(model, loader, device, num_classes):
    model.eval()
    correct = 0
    total = 0
    for data in loader:
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=1)
        y = data.y.view(-1, num_classes).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += data.num_graphs
    return correct / total

def class_conditional_lambda(class_idx, base_range, counts):
    c = counts[class_idx]
    max_c = counts.max()
    scale = (max_c / c).clamp(1.0, 5.0)
    low, high = base_range
    return np.random.uniform(low, high * scale.item())

def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = list(TUDataset(root="data", name=args.dataset))
    dataset = drop_edge_attributes(dataset)

    for g in dataset:
        g.y = g.y.view(-1)

    dataset = prepare_dataset_onehot_y(dataset)

    model_fns = {
        "GIN": lambda nf, nc: GIN(nf, nc, args.hidden_dim),
        "GCN": lambda nf, nc: GCN(nf, nc, args.hidden_dim),
        "TopKPool": lambda nf, nc: TopKPool(nf, nc, args.hidden_dim),
        "GCN2": lambda nf, nc: GCN2(
            nf,
            nc,
            hidden_dim=args.hidden_dim,
            num_layers=2,
            alpha=0.1,
            theta=0.5,
            dropout=0.3
        )
    }

    results = {name: [] for name in model_fns}

    for seed in args.seeds:
        print(f"\n=== Seed {seed} ===")
        random.seed(seed)
        torch.manual_seed(seed)

        random.shuffle(dataset)

        n = len(dataset)
        train_n = int(0.7 * n)
        val_n = int(0.8 * n)

        base_train_set = dataset[:train_n]

        counts, freqs = get_class_stats(base_train_set)
        print("Class counts:", counts.tolist())
        print("Imbalance ratio:", (counts.max() / counts.min()).item())

        class_weights = (1.0 / counts).to(device)
        class_weights /= class_weights.sum()

        class_graphs = split_class_graphs(base_train_set)

        _, _, _, median_nodes, _, _ = stat_graph(base_train_set)
        resolution = int(median_nodes)

        graphons = []
        for label, graphs in class_graphs:
            aligned, _, _, _ = align_graphs(
                graphs, padding=True, N=resolution
            )
            graphon = universal_svd(aligned, threshold=0.2)
            graphons.append((label, graphon))

        labels = torch.tensor(
            [onehot_to_index(label) for label, _ in graphons],
            dtype=torch.long
        )

        label_counts = counts.index_select(0, labels)
        weights = 1.0 / label_counts.float()
        weights /= weights.sum()

        num_sample = int(train_n * args.aug_ratio / args.aug_num)

        new_graphs = []
        for _ in range(args.aug_num):
            idxs = torch.multinomial(weights, 2, replacement=False)
            g1 = graphons[idxs[0]]
            g2 = graphons[idxs[1]]

            class_idx = onehot_to_index(g1[0])
            lam = class_conditional_lambda(
                class_idx, args.lam_range, counts
            )

            new_graphs += two_graphons_mixup(
                [g1, g2], la=lam, num_sample=num_sample
            )

        full_dataset = drop_edge_attributes(new_graphs + dataset)
        full_dataset = prepare_dataset_x(full_dataset)

        num_features = full_dataset[0].x.shape[1]
        num_classes = full_dataset[0].y.shape[0]

        train_set = full_dataset[: train_n + len(new_graphs)]
        test_set = full_dataset[val_n + len(new_graphs):]

        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True
        )
        test_loader = DataLoader(
            test_set, batch_size=args.batch_size
        )

        for name, build in model_fns.items():
            print(f"Training {name}")
            model = build(num_features, num_classes).to(device)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=args.lr, weight_decay=5e-4
            )

            for _ in range(args.epochs):
                train_epoch(
                    model,
                    train_loader,
                    optimizer,
                    device,
                    num_classes,
                    class_weights,
                )

            acc = eval_model(model, test_loader, device, num_classes)
            results[name].append(acc)
            print(f"{name} acc: {acc:.4f}")

    return results

def print_table(results):
    print("\n=== GMixup (Imbalance-Aware) Results ===")
    print("--------------------------------------")
    print(f"{'Model':<12} | {'Mean':<8} | {'Std':<8}")
    print("--------------------------------------")
    for model, accs in results.items():
        print(f"{model:<12} | {np.mean(accs):.4f} | {np.std(accs):.4f}")


def save_table(results, dataset, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{dataset}_gmixup_imbalance.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Mean", "Std"])
        for model, accs in results.items():
            writer.writerow([
                model,
                f"{np.mean(accs):.4f}",
                f"{np.std(accs):.4f}",
            ])
    print(f"\nSaved results to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--aug_ratio", type=float, default=0.15)
    parser.add_argument("--aug_num", type=int, default=10)
    parser.add_argument("--lam_range", type=float, nargs=2, default=[0.005, 0.01])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--out_dir", type=str, default="results")

    args = parser.parse_args()

    results = run_experiment(args)
    print_table(results)
    save_table(results, args.dataset, args.out_dir)
