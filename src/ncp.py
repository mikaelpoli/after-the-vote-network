""" SETUP """
# LIBRARIES
from pathlib import Path
import igraph as ig
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import sys


# DIRECTORIES
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path().resolve()

SRC_DIR = BASE_DIR / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

DATA_DIR = BASE_DIR / 'data'
DATA_DIR.mkdir(exist_ok=True)
POSTS_DIR = DATA_DIR / 'posts'
POSTS_ALL_DIR = POSTS_DIR / 'all'
POSTS_FILTERED_DIR = POSTS_DIR / 'filtered'
COMMENTS_DIR = DATA_DIR / 'comments'
RESULTS_DIR = BASE_DIR / 'results'


""" FUNCTIONS """
def threshold_graph(graph, percentile=90):
    weights = graph.es["weight"]
    threshold = np.percentile(weights, percentile)
    edges_to_keep = [e.index for e in graph.es if e["weight"] >= threshold]
    return graph.subgraph_edges(edges_to_keep, delete_vertices=False)


def conductance(graph, subset, sources, targets, weights):
    in_subset = np.zeros(graph.vcount(), dtype=bool)
    in_subset[subset] = True

    in_S = in_subset[sources]
    in_S_target = in_subset[targets]
    cut_mask = np.logical_xor(in_S, in_S_target)

    cut_size = weights[cut_mask].sum()

    vol_S = sum(graph.strength(subset, weights="weight"))
    vol_rest = sum(graph.strength(weights="weight")) - vol_S

    return cut_size / min(vol_S, vol_rest) if vol_S and vol_rest else 1


def process_seed(graph, seed, k, sources, targets, weights):
    pr = graph.personalized_pagerank(reset_vertices=[seed], damping=0.85)
    sorted_nodes = np.argsort(pr)[::-1]
    comm = sorted_nodes[:k].tolist()
    phi = conductance(graph, comm, sources, targets, weights)
    return phi


def compute_ncp(graph, sizes, num_seeds=10, n_jobs=-1, verbose=True):
    np.random.seed(42)
    ncp = []

    # Precompute edge arrays once
    sources = np.array([e.source for e in graph.es])
    targets = np.array([e.target for e in graph.es])
    weights = np.array(graph.es["weight"])

    for k in sizes:
        seeds = np.random.choice(graph.vs.indices, num_seeds, replace=False)

        # Parallel computation
        phis = Parallel(n_jobs=n_jobs)(
            delayed(process_seed)(graph, seed, k, sources, targets, weights)
            for seed in seeds
        )

        best_phi = min(phis)
        ncp.append(best_phi)

        if verbose:
            print(f"Size {k}: best conductance = {best_phi:.4f}")

    return ncp


def plot_ncp(sizes, ncp_data, thresholds, title="Network Community Profile for Different Thresholds"):
    """
    Parameters:
    - sizes: list or array of community sizes.
    - ncp_data: list of lists/arrays, each representing conductance values for a threshold.
    - thresholds: list of threshold identifiers (e.g., [90, 95, 99]).
    """
    plt.figure(figsize=(8, 5))
    
    for ncp, threshold in zip(ncp_data, thresholds):
        label = f"{threshold}th percentile threshold"
        plt.plot(sizes, ncp, label=label)
    
    plt.xlabel("Community Size")
    plt.ylabel("Minimum Conductance")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()