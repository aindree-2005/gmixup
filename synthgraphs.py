import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
import argparse

def sample_graph_from_stepfunc(stepfunc: np.ndarray,label: np.ndarray,num_sample: int = 1,
):
   
    sample_graphs = []
    sample_graph_label = torch.from_numpy(label).float()

    for _ in range(num_sample):
        # Bernoulli sampling
        A = (np.random.rand(*stepfunc.shape) <= stepfunc).astype(np.int32)

        # Symmetrize
        A = np.triu(A)
        A = A + A.T - np.diag(np.diag(A))

        # Remove isolated nodes
        A = A[A.sum(axis=1) != 0]
        A = A[:, A.sum(axis=0) != 0]

        if A.size == 0:
            continue

        A = torch.from_numpy(A)
        edge_index, _ = dense_to_sparse(A)

        num_nodes = int(torch.max(edge_index)) + 1

        pyg_graph = Data()
        pyg_graph.y = sample_graph_label
        pyg_graph.edge_index = edge_index
        pyg_graph.num_nodes = num_nodes

        sample_graphs.append(pyg_graph)

    return sample_graphs
def stepfunc_mixup(S1: np.ndarray,S2: np.ndarray,y1: np.ndarray,y2: np.ndarray,la: float,
):
    """
    Mix two step-functions and labels (as in utils.py)
    """
    mixed_stepfunc = la * S1 + (1 - la) * S2
    mixed_label = la * y1 + (1 - la) * y2
    return mixed_stepfunc, mixed_label
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_adj_matrix(A, title="", save_path=None):
    """
    Plot adjacency matrix of a synthetic graph
    """
    if torch.is_tensor(A):
        A = A.cpu().numpy()

    plt.figure(figsize=(4, 4))
    plt.imshow(A, cmap="gray_r")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
import networkx as nx


def plot_graph_structure(data, title="", node_size=40):
    """
    Plot PyG graph using NetworkX
    """
    edge_index = data.edge_index.cpu().numpy()
    G = nx.Graph()
    G.add_edges_from(edge_index.T.tolist())

    plt.figure(figsize=(4, 4))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G,
        pos,
        node_size=node_size,
        width=0.5,
        alpha=0.8
    )
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def main(args):
    if args.out_dir is None:
        stepfunc_dir = f"{args.dataset}_results"
    else:
        stepfunc_dir = args.out_dir
    print(f"Loading step-functions from: {stepfunc_dir}")
    stepfunc_files = sorted([
        f for f in os.listdir(stepfunc_dir)
        if f.startswith("class_") and f.endswith("_stepfunc.npy")
    ])

    num_classes = len(stepfunc_files)
    print(f"Found {num_classes} classes")

    synthetic_by_class = {}

    for cls in range(num_classes):
        stepfunc_path = os.path.join(
            stepfunc_dir,
            f"class_{cls}_stepfunc.npy"
        )

        if not os.path.exists(stepfunc_path):
            print(f"Skipping class {cls}: stepfunc not found")
            continue

        stepfunc = np.load(stepfunc_path)

        # one-hot label
        label = np.zeros(num_classes)
        label[cls] = 1.0

        graphs = sample_graph_from_stepfunc(
            stepfunc=stepfunc,
            label=label,
            num_sample=args.num_sample
        )

        synthetic_by_class[cls] = graphs
        print(f"Class {cls}: generated {len(graphs)} synthetic graphs")

        # optional visualization
        if args.visualize and len(graphs) > 0:
            plot_graph_structure(
                graphs[0],
                title=f"Synthetic graph – class {cls}"
            )

    print("Synthetic graph generation finished.")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["PROTEINS", "IMDB-BINARY", "REDDIT-BINARY", "IMDB-MULTI"]
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Directory containing class_*_stepfunc.npy"
    )

    parser.add_argument(
        "--num_sample",
        type=int,
        default=20,
        help="Number of synthetic graphs per class"
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize one synthetic graph per class"
    )

    args = parser.parse_args()
    main(args)
