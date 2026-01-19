import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_dense_adj

from graphonestimator import estimate_graphon

def load_dataset(name, root="data"):
    return TUDataset(root=root, name=name)

def pyg_graph_to_adj(data):
    """Convert PyG graph to dense adjacency matrix"""
    adj = to_dense_adj(
        data.edge_index,
        max_num_nodes=data.num_nodes
    ).to(torch.float32)
    return adj[0].cpu().numpy()

def group_graphs_by_class(dataset):
    class_graphs = {}
    for data in dataset:
        label = int(data.y.item())
        adj = pyg_graph_to_adj(data)
        if label not in class_graphs:
            class_graphs[label] = []
        class_graphs[label].append(adj)
    return class_graphs
def plot_graphon(graphon, title, save_path):
    """
    Plot a graphon
    """
    plt.figure(figsize=(5, 5))
    im = plt.imshow(
        graphon,
        cmap="plasma",
        origin="upper",     
        vmin=0.0,
        vmax=1.0,
        extent=[0, 1, 0, 1] # latent space [0,1] × [0,1]
    )
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xlabel("Latent node position (u)")
    plt.ylabel("Latent node position (v)")
    plt.title(title)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
def plot_stepfunc(stepfunc, title, save_path):
    """
    Plot step-function graphon 
    """
    plt.figure(figsize=(5, 5))
    im = plt.imshow(
        stepfunc,
        cmap="plasma",
        origin="upper",
        vmin=0.0,
        vmax=1.0,
        extent=[0, 1, 0, 1],
        interpolation="nearest"  
    )
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xlabel("Latent node position (u)")
    plt.ylabel("Latent node position (v)")
    plt.title(title)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def main(args):
    if args.out_dir is None:
        args.out_dir = f"{args.dataset}_results"
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset)
    node_counts = [data.num_nodes for data in dataset]
    print("Determining global N...")
    if args.N is not None: #user-specified
        N_global = args.N
        print(f"Using user-specified N = {N_global}")
    else: # automatic selection from full dataset
        if args.N_mode == "median":
            N_global = int(np.median(node_counts))
        else:
            N_global = int(np.mean(node_counts))
        print(f"Using {args.N_mode} N = {N_global} (computed from dataset)")
    args.N = N_global
    print("Grouping graphs by class...")
    class_graphs = group_graphs_by_class(dataset)

    print(f"Found {len(class_graphs)} classes")
    for cls, graphs in class_graphs.items():
        print(f"Estimating graphon for class {cls} with {len(graphs)} graphs")

        stepfunc, graphon = estimate_graphon(
            graphs,
            method=args.method,
            args=args
        )

        np.save(os.path.join(args.out_dir, f"class_{cls}_stepfunc.npy"), stepfunc)
        np.save(os.path.join(args.out_dir, f"class_{cls}_graphon.npy"), graphon)

        plot_graphon(
            graphon,
            title=f"{args.dataset} – Class {cls}",
            save_path=os.path.join(args.out_dir, f"class_{cls}_graphon.png")
        )
        plot_stepfunc(
            stepfunc,
            title=f"{args.dataset} – Class {cls} (step)",
            save_path=os.path.join(args.out_dir, f"class_{cls}_stepfunc.png")
        )


    print("Finished graphon estimation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True,
                        choices=["PROTEINS", "IMDB-BINARY", "REDDIT-BINARY","IMDB-MULTI"])
    parser.add_argument("--method", type=str, default="USVT")
    parser.add_argument("--threshold_usvt", type=float, default=1e-6)
    parser.add_argument("--r", type=int, default=32,
                        help="Resolution of graphon")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument(
        "--N",
        type=int,
        default=None,
        help="Target number of nodes (overrides automatic selection)"
    )

    parser.add_argument(
        "--N_mode",
        type=str,
        choices=["median", "mean"],
        default="median",
        help="How to choose N when --N is not provided"
    )
        
    args = parser.parse_args()
    main(args)