""" SETUP """
# LIBRARIES
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sys
import time


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

def build_adjacency_from_embeddings(embeddings, threshold=0.3):
    sim_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(sim_matrix, 0)  # Remove self-similarity
    A = (sim_matrix > threshold).astype(float) * sim_matrix  # Thresholding
    return A


def compute_P_dd(A):
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # prevent division by zero
    return A / row_sums


def compute_P_tt(P_dd, C):
    return C.T @ P_dd @ C


def compute_p_t(P_tt):
    return P_tt @ np.ones((P_tt.shape[0], 1))


def compute_modularity(P_tt, p_t):
    return P_tt.trace() - np.dot(p_t.flatten(), p_t.flatten())