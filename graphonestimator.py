#reference source: https://github.com/ahxt/g-mixup/blob/master/src/graphon_estimator.py
import copy
import cv2
import numpy as np
import torch
#from skimage.restoration import denoise_tv_chambolle
from typing import List, Tuple

def graph_numpy2tensor(graphs: List[np.ndarray]) -> torch.Tensor:
    """
    Convert a list of np arrays to a pytorch tensor
    :param graphs: [K (N, N) adjacency matrices]
    :return:
        graph_tensor: [K, N, N] tensor
    """
    graph_tensor = np.array(graphs)
    return torch.from_numpy(graph_tensor).float()
def align_graphs(graphs: List[np.ndarray],
                 padding: bool = False, N: int =None) -> Tuple[List[np.ndarray], List[np.ndarray], int, int]:
    """
    Align multiple graphs by sorting their nodes by descending node degrees

    :param graphs: a list of binary adjacency matrices
    :param padding: whether padding graphs to the same size or not
    :return:
        aligned_graphs: a list of aligned adjacency matrices
        normalized_node_degrees: a list of sorted normalized node degrees (as node distributions)
    """
    MAX_N=512 #numpy memory issue thing
    num_nodes = [g.shape[0] for g in graphs]
    max_num = max(num_nodes) if max(num_nodes)<=MAX_N else MAX_N
    min_num = min(num_nodes)
    if N is not None:
        max_num = max(max_num, N)

    aligned_graphs = []
    normalized_node_degrees = []

    for i in range(len(graphs)):
        num_i = graphs[i].shape[0] if graphs[i].shape[0]<=MAX_N else MAX_N
        node_degree = 0.5 * np.sum(graphs[i], axis=0) + 0.5 * np.sum(graphs[i], axis=1)
        node_degree /= np.sum(node_degree)

        idx = np.argsort(node_degree)[::-1]

        sorted_node_degree = node_degree[idx].reshape(-1, 1)
        sorted_graph = graphs[i][idx][:, idx]
        use_n = min(sorted_graph.shape[0], max_num)
        sorted_node_degree = sorted_node_degree[:use_n]
        sorted_graph = sorted_graph[:use_n, :use_n]

        if padding:
            normalized_node_degree = np.zeros((max_num, 1))
            normalized_node_degree[:num_i] = sorted_node_degree 

            aligned_graph = np.zeros((max_num, max_num))
            aligned_graph[:num_i, :num_i] = sorted_graph
        else:
            normalized_node_degree = sorted_node_degree
            aligned_graph = sorted_graph

        normalized_node_degrees.append(normalized_node_degree)
        aligned_graphs.append(aligned_graph)

    return aligned_graphs, normalized_node_degrees, max_num, min_num
def estimate_target_distribution(probs: List[np.ndarray], dim_t: int = None) -> np.ndarray:
    """
    Estimate target distribution via the average of sorted source probabilities
    Args:
        probs: a list of node distributions [(n_s, 1) the distribution of source nodes]
        dim_t: the dimension of target distribution
    Returns:
        p_t: (dim_t, 1) vector representing a distribution
    """
    if dim_t is None:
        dim_t = min([probs[i].shape[0] for i in range(len(probs))])

    p_t = np.zeros((dim_t, 1))
    x_t = np.linspace(0, 1, p_t.shape[0])
    for i in range(len(probs)):
        p_s = probs[i][:, 0]
        p_s = np.sort(p_s)[::-1]
        x_s = np.linspace(0, 1, p_s.shape[0])
        p_t_i = np.interp(x_t, x_s, p_s) + 1e-3
        p_t[:, 0] += p_t_i

    p_t /= np.sum(p_t)
    return p_t
def universal_svd(aligned_graphs: List[np.ndarray], threshold: float = 2.02) -> np.ndarray:
    """
    Estimate a graphon by universal singular value thresholding.

    Reference:
    Chatterjee, Sourav.
    "Matrix estimation by universal singular value thresholding."
    The Annals of Statistics 43.1 (2015): 177-214.

    :param aligned_graphs: a list of (N, N) adjacency matrices
    :param threshold: the threshold for singular values
    :return: graphon: the estimated (r, r) graphon model
    """
    aligned_graphs = graph_numpy2tensor(aligned_graphs)
    num_graphs = aligned_graphs.size(0)

    if num_graphs > 1:
        sum_graph = torch.mean(aligned_graphs, dim=0)
    else:
        sum_graph = aligned_graphs[0, :, :]  # (N, N)

    num_nodes = sum_graph.size(0)

    u, s, v = torch.svd(sum_graph)
    singular_threshold = threshold * (num_nodes ** 0.5)
    binary_s = torch.lt(s, singular_threshold)
    s[binary_s] = 0
    graphon = u @ torch.diag(s) @ torch.t(v)
    graphon[graphon > 1] = 1
    graphon[graphon < 0] = 0
    graphon = graphon.numpy()
    return graphon

def estimate_graphon(graphs: List[np.ndarray], method, args):
    if method in ['GWB', 'SGWB', 'FGWB', 'SFGWB']:
        aligned_graphs, normalized_node_degrees, max_num, min_num = align_graphs(
            graphs, padding=False, N=args.N
        )
    else:
        aligned_graphs, normalized_node_degrees, max_num, min_num = align_graphs(
            graphs, padding=True, N=args.N
        )
    if args.N is not None:
        N = args.N
        aligned_graphs = [A[:N, :N] for A in aligned_graphs]
        normalized_node_degrees = [d[:N] for d in normalized_node_degrees]
    else:
        N = max_num

    block_size = int(np.log2(N) + 1)
    num_blocks = int(N / block_size)

    p_b = estimate_target_distribution(normalized_node_degrees, dim_t=num_blocks)

    stepfunc = universal_svd(aligned_graphs, threshold=args.threshold_usvt)
    graphon = cv2.resize(stepfunc, (args.r, args.r), interpolation=cv2.INTER_LINEAR)

    return stepfunc, graphon
