import os
import argparse
import random
import csv
import copy
import numpy as np
import torch
import torch.nn.functional as F

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

def train_epoch(model, loader, optimizer, device):
    model.train()
    loss_all = 0
    graph_all = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()

        loss_all += loss.item() * data.num_graphs
        graph_all += data.num_graphs

    return loss_all / graph_all


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    for data in loader:
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=1)
        correct += pred.eq(data.y).sum().item()
        total += data.num_graphs

    return correct / total


def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TUDataset(root="data", name=args.dataset)
    dataset = list(dataset)

    for g in dataset:
        g.y = g.y.view(-1)

    dataset = prepare_dataset_x(dataset)

    num_features = dataset[0].x.shape[1]
    num_classes = int(max(g.y.item() for g in dataset)) + 1

    model_fns = {
        "GIN": lambda: GIN(num_features, num_classes, args.hidden_dim),
        "GCN": lambda: GCN(num_features, num_classes, args.hidden_dim),
        "TopKPool": lambda: TopKPool(num_features, num_classes, args.hidden_dim),
        "GCN2": lambda: GCN2(
            num_features,
            num_classes,
            hidden_dim=args.hidden_dim,
            num_layers=2,
            alpha=0.1,
            theta=0.4,
            dropout=0.3,
        ),
    }

    results = {name: [] for name in model_fns}

    for seed in args.seeds:
        print(f"\n=== Seed {seed} ===")
        set_seed(seed)

        random.shuffle(dataset)
        n = len(dataset)

        train_end = int(0.8 * n)
        val_end = int(0.9 * n)

        train_dataset = dataset[:train_end]
        val_dataset = dataset[train_end:val_end]
        test_dataset = dataset[val_end:]

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

        for name, build_model in model_fns.items():
            print(f"Training {name}...")
            model = build_model().to(device)

            optimizer = torch.optim.Adam(
                model.parameters(), lr=args.lr, weight_decay=5e-4
            )

            best_val = 0.0
            best_state = None

            for _ in range(args.epochs):
                train_epoch(model, train_loader, optimizer, device)
                val_acc = eval_model(model, val_loader, device)

                if val_acc > best_val:
                    best_val = val_acc
                    best_state = copy.deepcopy(model.state_dict())

            model.load_state_dict(best_state)
            test_acc = eval_model(model, test_loader, device)

            results[name].append(test_acc)
            print(f"{name} test acc: {test_acc:.4f}")

    return results


def print_table(results):
    print("\n=== Vanilla Results ===")
    print("-----------------------")
    print(f"{'Model':<12} | {'Mean':<8} | {'Std':<8}")
    print("-----------------------")

    for model, accs in results.items():
        print(
            f"{model:<12} | {np.mean(accs):.4f} | {np.std(accs):.4f}"
        )


def save_table(results, dataset, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{dataset}_vanilla.csv")

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

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["PROTEINS", "IMDB-BINARY", "REDDIT-BINARY", "IMDB-MULTI", "MUTAG"],
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--out_dir", type=str, default="results")

    args = parser.parse_args()

    results = run_experiment(args)
    print_table(results)
    save_table(results, args.dataset, args.out_dir)
