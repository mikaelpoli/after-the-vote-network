""" SETUP """
# LIBRARIES
from pathlib import Path
import scipy.sparse as sps
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import os
import pandas as pd
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

def logg(x):
    y = np.log(x)
    y[x == 0] = 0

    return y


def build_C(topics, with_outliers=2):
    C = sps.csr_matrix((len(topics),topics.max()+with_outliers))
    for i in range(C.shape[1]):
        C[np.array(topics)==(i-1),i] = 1
    
    return C


def build_probability_matrices(Mwd, topics):
    Mwd = Mwd.astype(np.float64)

    # Remove outlier documents
    topics = np.array(topics)
    valid_idx = topics != -1
    topics = topics[valid_idx]
    Mwd = Mwd[:, valid_idx]
    
    n_words, n_docs = Mwd.shape
    n_topics = topics.max() + 1

    # Document-topic matrix
    rows = np.arange(n_docs)
    cols = topics
    data = np.ones(n_docs)
    C = sps.csr_matrix((data, (rows, cols)), shape=(n_docs, n_topics))

    # Compute P(w,d)
    Pwd = Mwd / Mwd.sum()

    # Compute P(w,c)
    Pwc = Pwd @ C
    Pwc = Pwc / Pwc.sum()

    # Compute P(d,d)
    pw = Pwd.sum(axis=1).flatten()  # p(w)
    pw[pw == 0] = 1                 # Avoid division by 0
    Pdd = (Pwd.T / pw) @ Pwd        # shape: (n_docs, n_docs)

    # Compute P(c,c)
    Pcc = C.T @ Pdd @ C

    return Pwc, Pdd, Pcc


def calculate_nmi(Pwc):
    aw = Pwc.sum(axis=1).flatten()  # Word marginal probs
    ac = Pwc.sum(axis=0).flatten()  # Topic marginal probs

    # Avoid division by zero
    aw[aw == 0] = 1e-12
    ac[ac == 0] = 1e-12

    Hc = np.multiply(ac, -logg(ac)).sum()  # Entropy of topics

    A2 = ((Pwc / ac).T / aw).T  # Joint over marginals
    A2.data = logg(A2.data)

    return (Pwc.multiply(A2)).sum() / Hc  # NMI


def calculate_ncut(A):
    row_sum = np.array(A.sum(axis=1)).flatten()
    col_sum = np.array(A.sum(axis=0)).flatten()
    cut_value = ((col_sum - A.diagonal()) / col_sum).mean()

    return cut_value


def calculate_modularity(Pcc):
    return Pcc.trace() - (Pcc.sum(axis=0) * Pcc.sum(axis=1)).item()


def save_results_to_csv(csv_path, metrics, metric_names, model_name):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0)
    else:
        df = pd.DataFrame(index=metric_names)

    # Ensure all metric rows exist (in case new ones were added)
    for metric in metrics[model_name].keys():
        if metric not in df.index:
            df.loc[metric] = np.nan

    # Add or update the model's column
    for metric, value in metrics[model_name].items():
        df.loc[metric, model_name] = value

    df.to_csv(csv_path)