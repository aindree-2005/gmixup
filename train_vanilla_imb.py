import os
import argparse
import random
import csv
import copy
import math
import numpy as np
import torch
import torch.nn.functional as F

from collections import Counter
from sklearn.metrics import f1_score, recall_score

from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.utils import degree

from models import GCN2, GIN, GCN, TopKPool


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_dataset_x(dataset):
    if dataset[0].x is None:
        max_degree = 0
        degs = []

        for data in dataset:
            deg = degree(data.edge_index[0], dtype=torch.long)
            degs.append(deg)
            max_degree = max(max_degree, deg.max().item())

        if max_degree < 2000:
            for data in dataset:
                deg = degree(data.edge_index[0], dtype=torch.long)
                data.x = F.one_hot(deg, num_classes=max_degree + 1).float()
        else:
            all_deg = torch.cat(degs).float()
            mean, std = all_deg.mean(), all_deg.std()
            for data in dataset:
                deg = degree(data.edge_index[0], dtype=torch.long)
                data.x = ((deg - mean) / std).view(-1, 1)

    return dataset
def split_data(dataset, args):
    labels = np.array([g.y.item() for g in dataset])
    classes = np.unique(labels)

    train_ds, val_ds, test_ds = [], [], []

    for c in classes:
        idx = np.where(labels == c)[0]
        np.random.shuffle(idx)

        # training: IMBALANCED
        ratio = args.imb_ratio[int(c)] / sum(args.imb_ratio)
        n_train_c = int(ratio * args.train_num)

        # validation + test: BALANCED
        n_val_c = args.val_num // len(classes)

        train_ds.extend([dataset[i] for i in idx[:n_train_c]])
        val_ds.extend([dataset[i] for i in idx[n_train_c:n_train_c + n_val_c]])
        test_ds.extend([dataset[i] for i in idx[n_train_c + n_val_c:]])

    # sanity checks
    assert len(set(g.y.item() for g in train_ds)) > 1
    assert len(set(g.y.item() for g in val_ds)) > 1
    assert len(set(g.y.item() for g in test_ds)) > 1

    return train_ds, val_ds, test_ds



def compute_class_weights(train_ds, num_classes, device):
    labels = [g.y.item() for g in train_ds]
    counts = Counter(labels)
    total = sum(counts.values())

    weights = torch.zeros(num_classes)
    for c in range(num_classes):
        weights[c] = total / counts.get(c, 1)

    return weights.to(device)


def upsample_dataset(train_ds):
    counts = Counter(g.y.item() for g in train_ds)
    max_count = max(counts.values())

    by_class = {}
    for g in train_ds:
        by_class.setdefault(g.y.item(), []).append(g)

    new_ds = []
    for graphs in by_class.values():
        reps = max_count // len(graphs)
        rem = max_count % len(graphs)
        new_ds.extend(graphs * reps)
        new_ds.extend(random.sample(graphs, rem))

    random.shuffle(new_ds)
    return new_ds

def train_epoch(model, loader, optimizer, device, loss_fn):
    model.train()
    total_loss = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        logits = model(data)              # RAW LOGITS
        loss = loss_fn(logits, data.y)    # CE expects logits
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    ys, preds = [], []

    for data in loader:
        data = data.to(device)
        logits = model(data)
        pred = logits.argmax(dim=1)
        ys.extend(data.y.cpu().numpy())
        preds.extend(pred.cpu().numpy())

    ys = np.array(ys)
    preds = np.array(preds)

    acc = (ys == preds).mean()
    f1_macro = f1_score(ys, preds, average="macro", zero_division=0)
    f1_micro = f1_score(ys, preds, average="micro", zero_division=0)
    recall_macro = recall_score(ys, preds, average="macro", zero_division=0)

    return acc, f1_macro, f1_micro, recall_macro
def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = list(TUDataset(root="data", name=args.dataset))
    for g in dataset:
        g.y = g.y.view(-1)

    dataset = prepare_dataset_x(dataset)

    num_features = dataset[0].x.size(1)
    num_classes = int(max(g.y.item() for g in dataset)) + 1

    model_fns = {
        "GIN": lambda: GIN(num_features, num_classes, args.hidden_dim),
        "GCN": lambda: GCN(num_features, num_classes, args.hidden_dim),
        "TopKPool": lambda: TopKPool(num_features, num_classes, args.hidden_dim),
        "GCN2": lambda: GCN2(
            num_features, num_classes,
            hidden_dim=args.hidden_dim,
            num_layers=2,
            alpha=0.1, theta=0.4, dropout=0.3,
        ),
    }

    settings = ["vanilla", "reweight", "upsample"]
    results = {
        s: {m: {"acc": [], "f1_macro": [], "f1_micro": [], "recall_macro": []}
            for m in model_fns}
        for s in settings
    }

    for seed in args.seeds:
        print(f"\n=== Seed {seed} ===")
        set_seed(seed)
        random.shuffle(dataset)

        train_ds, val_ds, test_ds = split_data(dataset, args)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size)

        up_train_loader = DataLoader(
            upsample_dataset(train_ds),
            batch_size=args.batch_size,
            shuffle=True
        )

        class_weights = compute_class_weights(train_ds, num_classes, device)

        for model_name, build_model in model_fns.items():
            for setting in settings:
                model = build_model().to(device)
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=args.lr, weight_decay=5e-4
                )

                if setting == "vanilla":
                    loader = train_loader
                    loss_fn = torch.nn.CrossEntropyLoss()
                elif setting == "reweight":
                    loader = train_loader
                    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
                else:
                    loader = up_train_loader
                    loss_fn = torch.nn.CrossEntropyLoss()

                best_val = -1
                best_state = None

                for _ in range(args.epochs):
                    train_epoch(model, loader, optimizer, device, loss_fn)
                    _, val_f1, _, _ = eval_model(model, val_loader, device)

                    if val_f1 > best_val:
                        best_val = val_f1
                        best_state = copy.deepcopy(model.state_dict())

                model.load_state_dict(best_state)
                acc, f1_macro, f1_micro, recall_macro = eval_model(
                    model, test_loader, device
                )

                results[setting][model_name]["acc"].append(acc)
                results[setting][model_name]["f1_macro"].append(f1_macro)
                results[setting][model_name]["f1_micro"].append(f1_micro)
                results[setting][model_name]["recall_macro"].append(recall_macro)

                print(f"{model_name} | {setting}: "
                      f"Acc={acc:.3f}, F1-macro={f1_macro:.3f}")

    return results


def save_results(results, dataset, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{dataset}_imbalance_compare.csv")

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Setting", "Model",
            "Acc_mean", "Acc_std",
            "F1_macro_mean", "F1_macro_std",
            "F1_micro_mean", "F1_micro_std",
            "Recall_macro_mean", "Recall_macro_std",
        ])

        for setting, models in results.items():
            for model, vals in models.items():
                writer.writerow([
                    setting, model,
                    np.mean(vals["acc"]), np.std(vals["acc"]),
                    np.mean(vals["f1_macro"]), np.std(vals["f1_macro"]),
                    np.mean(vals["f1_micro"]), np.std(vals["f1_micro"]),
                    np.mean(vals["recall_macro"]), np.std(vals["recall_macro"]),
                ])

    print(f"\nSaved results to {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="PROTEINS")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])

    parser.add_argument("--train_num", type=int, default=300)
    parser.add_argument("--val_num", type=int, default=300)
    parser.add_argument("--imb_ratio", nargs="+", type=float, default=[1, 9])

    parser.add_argument("--out_dir", type=str, default="results")

    args = parser.parse_args()

    results = run_experiment(args)
    save_results(results, args.dataset, args.out_dir)
