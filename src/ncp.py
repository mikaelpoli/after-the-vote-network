""" SETUP """
# LIBRARIES
from pathlib import Path
import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def conductance(graph, subset):
    subset_set = set(subset)
    cut_size = 0.0
    for v in subset:
        for e in graph.es.select(_source=v):
            if e.target not in subset_set:
                cut_size += e["weight"]
    vol_S = sum(graph.strength(subset))  # Sum of weighted degrees
    vol_rest = sum(graph.strength()) - vol_S
    return cut_size / min(vol_S, vol_rest) if vol_S and vol_rest else 1


def local_pagerank_sweep(graph, seed, alpha=0.85):
    pr = graph.personalized_pagerank(reset_vertices=[seed], damping=alpha)
    sorted_nodes = np.argsort(pr)[::-1]
    return [sorted_nodes[:k].tolist() for k in range(1, len(sorted_nodes)+1)]


def compute_ncp(graph, sizes, num_seeds=10):
    ncp = []
    for k in sizes:
        best_phi = 1.0
        np.random.seed(42)
        seeds = np.random.choice(graph.vs.indices, num_seeds, replace=False)
        for seed in seeds:
            for comm in local_pagerank_sweep(graph, seed):
                if len(comm) >= k:
                    phi = conductance(graph, comm[:k])
                    if phi < best_phi:
                        best_phi = phi
                    break
        ncp.append(best_phi)
    return ncp


def plot_ncp(ncp_90, ncp_95, ncp_99, sizes):
    plt.figure(figsize=(8, 5))
    plt.plot(sizes, ncp_90, label="90th percentile threshold")
    plt.plot(sizes, ncp_95, label="95th percentile threshold")
    plt.plot(sizes, ncp_99, label="99th percentile threshold")
    plt.xlabel("Community Size")
    plt.ylabel("Minimum Conductance")
    plt.title("Network Community Profile for Different Thresholds")
    plt.legend()
    plt.grid(True)
    plt.show()