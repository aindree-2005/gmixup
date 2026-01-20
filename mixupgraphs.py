import os
import argparse
import numpy as np
import random

from synthgraphs import (
    sample_graph_from_stepfunc,
    stepfunc_mixup,
    plot_graph_structure
)

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
    stepfuncs = {}
    labels = {}

    for cls in range(num_classes):
        path = os.path.join(stepfunc_dir, f"class_{cls}_stepfunc.npy")
        stepfuncs[cls] = np.load(path)

        y = np.zeros(num_classes, dtype=np.float32)
        y[cls] = 1.0
        labels[cls] = y
    random.seed(args.seed)
    mixup_graphs = []

    for i in range(args.num_mixup):
        c1, c2 = random.sample(range(num_classes), 2)
        lam = np.random.uniform(args.lam_min, args.lam_max)

        print(f"Mix {i}: class {c1} + class {c2}, λ={lam:.4f}")

        S_mix, y_mix = stepfunc_mixup(
            stepfuncs[c1],
            stepfuncs[c2],
            labels[c1],
            labels[c2],
            lam
        )

        graphs = sample_graph_from_stepfunc(
            stepfunc=S_mix,
            label=y_mix,
            num_sample=1
        )

        if len(graphs) > 0:
            mixup_graphs.append(graphs[0])

    print(f"Generated {len(mixup_graphs)} mixup graphs")

    if args.visualize:
        for i, g in enumerate(mixup_graphs[:args.max_show]):
            plot_graph_structure(
                g,
                title=f"Mixup Graph {i}"
            )

    print("Done.")

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

    parser.add_argument("--num_mixup", type=int, default=10)
    parser.add_argument("--lam_min", type=float, default=0.1)
    parser.add_argument("--lam_max", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize mixup graphs"
    )

    parser.add_argument(
        "--max_show",
        type=int,
        default=5,
        help="Max number of graphs to visualize"
    )

    args = parser.parse_args()
    main(args)
