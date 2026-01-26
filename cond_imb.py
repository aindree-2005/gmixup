import os
import csv
import random
import argparse
import numpy as np
from collections import Counter
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
import random
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
from sklearn.metrics import f1_score, recall_score


def onehot_to_index(label):
    if torch.is_tensor(label) and label.numel() > 1:
        return int(torch.argmax(label).item())
    return int(label)


def prepare_dataset_onehot_y(dataset):
    num_classes = len(set(int(g.y) for g in dataset))
    for g in dataset:
        g.y = F.one_hot(g.y.view(-1), num_classes=num_classes).float()[0]
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


def split_proteins_fixed(dataset,
                         train_min=30,
                         train_maj=270,
                         val_per_class=150,
                         seed=0):
    random.seed(seed)

    labels = [onehot_to_index(g.y) for g in dataset]
    class0 = [g for g, y in zip(dataset, labels) if y == 0]
    class1 = [g for g, y in zip(dataset, labels) if y == 1]

    random.shuffle(class0)
    random.shuffle(class1)

    if len(class0) <= len(class1):
        minority, majority = class0, class1
    else:
        minority, majority = class1, class0

    train_set = (
        minority[:train_min] +
        majority[:train_maj]
    )

    val_set = (
        minority[train_min:train_min + val_per_class] +
        majority[train_maj:train_maj + val_per_class]
    )

    test_set = (
        minority[train_min + val_per_class:] +
        majority[train_maj + val_per_class:]
    )

    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)

    return train_set, val_set, test_set
def ensure_minority(
    train_set,
    target_min=30,
    lam_range=(0.7, 0.95),
    resolution=400,
):
    labels = [onehot_to_index(g.y) for g in train_set]
    counts = Counter(labels)

    min_class = min(counts, key=counts.get)
    maj_class = max(counts, key=counts.get)

    deficit = target_min - counts[min_class]
    if deficit <= 0:
        return train_set

    minority_graphs = [g for g in train_set if onehot_to_index(g.y) == min_class]
    majority_graphs = [g for g in train_set if onehot_to_index(g.y) == maj_class]

    min_label, min_graphs = split_class_graphs(minority_graphs)[0]
    maj_label, maj_graphs = split_class_graphs(majority_graphs)[0]

    min_aligned, _, _, _ = align_graphs(min_graphs, padding=True, N=resolution)
    maj_aligned, _, _, _ = align_graphs(maj_graphs, padding=True, N=resolution)

    min_graphon = universal_svd(min_aligned, threshold=0.2)
    maj_graphon = universal_svd(maj_aligned, threshold=0.2)

    synthetic = []
    for _ in range(deficit):
        lam = np.random.uniform(*lam_range)
        synthetic += two_graphons_mixup(
            [(min_label, min_graphon), (min_label, maj_graphon)],
            la=lam,
            num_sample=1,
        )

    return train_set + synthetic

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
    y_true, y_pred = [], []

    for data in loader:
        data = data.to(device)
        out = model(data)
        y_pred.extend(out.argmax(dim=1).cpu().numpy())
        y_true.extend(
            data.y.view(-1, num_classes).argmax(dim=1).cpu().numpy()
        )

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return (
        (y_true == y_pred).mean(),
        f1_score(y_true, y_pred, average="micro"),
        f1_score(y_true, y_pred, average="macro"),
    )


def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print  ("Entered RUN EXPERIMENT")
    dataset = list(TUDataset(root="data", name="PROTEINS"))
    for g in dataset:
        g.y = g.y.view(-1)

    dataset = prepare_dataset_onehot_y(dataset)
    dataset = prepare_dataset_x(dataset)

    model_fns = {
        "GIN": lambda nf, nc: GIN(nf, nc, args.hidden_dim),
        "GCN": lambda nf, nc: GCN(nf, nc, args.hidden_dim),
        "TopKPool": lambda nf, nc: TopKPool(nf, nc, args.hidden_dim),
        #"GCN2": lambda nf, nc: GCN2(nf, nc, args.hidden_dim),
    }

    results = {name: [] for name in model_fns}

    for seed in args.seeds:
        print(f"\n=== Seed {seed} ===")
        random.seed(seed)
        torch.manual_seed(seed)

        train_set, val_set, test_set = split_proteins_fixed(dataset, seed=seed)
        train_set = ensure_minority(train_set)

        counts = Counter(onehot_to_index(g.y) for g in train_set)
        class_weights = torch.tensor(
            [1.0 / counts[i] for i in range(len(counts))]
        ).to(device)
        class_weights /= class_weights.sum()

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size)
        test_loader = DataLoader(test_set, batch_size=args.batch_size)

        nf = train_set[0].x.shape[1]
        nc = train_set[0].y.shape[0]

        for name, build in model_fns.items():
            print(f"Training {name}")
            model = build(nf, nc).to(device)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=args.lr, weight_decay=5e-4
            )

            best_f1 = 0
            best_metrics = None

            for _ in range(args.epochs):
                train_epoch(
                    model, train_loader, optimizer,
                    device, nc, class_weights
                )
                _, _, f1_macro = eval_model(
                    model, val_loader, device, nc
                )
                if f1_macro > best_f1:
                    best_f1 = f1_macro
                    best_metrics = eval_model(
                        model, test_loader, device, nc
                    )

            results[name].append(best_metrics)
            print(f"{name} | Acc {best_metrics[0]:.4f} | "
                  f"F1-micro {best_metrics[1]:.4f} | "
                  f"F1-macro {best_metrics[2]:.4f}")

    return results

def save_results(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "PROTEINS_fixed_split.csv")

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Model", "Seed", "Accuracy", "F1_micro", "F1_macro"]
        )
        for model, runs in results.items():
            for seed, (acc, f1_micro, f1_macro) in enumerate(runs):
                writer.writerow([
                    model, seed,
                    f"{acc:.4f}",
                    f"{f1_micro:.4f}",
                    f"{f1_macro:.4f}",
                ])

    print(f"\nSaved results to {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--out_dir", type=str, default="results")

    args = parser.parse_args()

    results = run_experiment(args)
    save_results(results, args.out_dir)
