""" SETUP """
# LIBRARIES
from bs4 import BeautifulSoup
from IPython.display import HTML
from pathlib import Path
from scipy.sparse import csr_matrix, find, lil_matrix
from sklearn.metrics.pairwise import cosine_similarity
import igraph as ig
import leidenalg
import matplotlib as mpl
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import re
import scipy.sparse as sps
import seaborn as sns
import spacy
import string
import sys
import time
import umap


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

def get_representative_topics(n_repr, bert_model, df, docs_col, topics_col, rank_col):
    doc_vectors = bert_model.vectorizer_model.transform(df[docs_col])
    topic_vectors = bert_model.c_tf_idf_
    similarity_matrix = cosine_similarity(doc_vectors, topic_vectors)

    representative_docs_idx = {}

    topics = df[topics_col].values
    df[rank_col] = np.nan

    for topic in np.unique(topics):
        idxs = np.where(topics == topic)[0]
        topic_similarities = similarity_matrix[idxs, topic]
        sorted_idx = idxs[np.argsort(topic_similarities)[::-1]]
        for rank, doc_idx in enumerate(sorted_idx, start=1):
            df.at[doc_idx, rank_col] = rank
        top_n = sorted_idx[:n_repr]
        representative_docs_idx[topic] = top_n
    
    return representative_docs_idx


def top_docs_per_topic(n_repr, df, og_text_col, topic_col, rank_col):
    top_docs_per_topic = {}

    for topic in df[topic_col].unique():
        if topic == -1:
            continue  # skip outliers
        top_docs = df[df[topic_col] == topic].sort_values(rank_col).head(n_repr)[og_text_col]
        top_docs_per_topic[topic] = top_docs.tolist()

    return top_docs_per_topic


def extract_community_assignments(bert_model_topics, topic_ids_start=-1):
    topic_assignments = bert_model_topics
    num_docs = len(topic_assignments)
    num_topics = max(topic_assignments) + topic_ids_start  # Topic IDs start from -1

    C = lil_matrix((num_docs, num_topics + topic_ids_start))  # +1 to handle topic -1
    for i in range(num_topics + 1):
        C[np.array(topic_assignments) == (i - 1), i] = 1
    C = C.tocsr()

    # Remove zero columns
    C = C[:, np.unique(find(C)[1])]

    return num_docs, num_topics, C


def plot_topic_pattern(num_docs, C):
    # Order topics
    topic_of_doc = C.argmax(axis=1).A1
    order = np.argsort(topic_of_doc)

    # Count docs per topic in ordered array:
    _, counts = np.unique(topic_of_doc[order], return_counts=True)

    # Get relative sizes
    rel_sizes = counts / counts.sum()
    _, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Left plot: topic size distribution
    ax[0].plot(np.arange(len(rel_sizes)), rel_sizes, 'o-', color='blue')
    ax[0].set_title('BERTopic Relative cluster sizes')
    ax[0].set_xlabel('Topic index')
    ax[0].set_ylabel('Proportion of docs')
    ax[0].grid(True)

    # Right plot: block diagonal matrix pattern
    cmap = plt.cm.viridis
    ax[1].set_facecolor(cmap(0))
    start = 0
    for size in counts:
        ax[1].fill_betweenx([start, start + size], start, start + size, color=cmap(1.0))
        start += size

    ax[1].set_xlim(0, num_docs)
    ax[1].set_ylim(0, num_docs)
    ax[1].invert_yaxis()
    ax[1].set_aspect('equal')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_title('BERTopic Cluster block pattern')

    plt.tight_layout()
    plt.show()