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


def plot_topic_network(bert_model, topic_labels_dict):
    # Filter out -1 topic and get consistent topic ids
    topic_info = bert_model.get_topic_info().sort_values("Topic")
    topic_info = topic_info[topic_info["Topic"] != -1]
    topic_ids = topic_info["Topic"].tolist()

    # Get embeddings and other info in correct order
    topic_embeddings = np.array([bert_model.topic_embeddings_[tid] for tid in topic_ids])
    topic_sizes = topic_info["Count"].values
    topic_labs = [topic_labels_dict[tid] for tid in topic_ids]
    vertex_sizes = 0.6 * topic_sizes
    t_colors = sns.color_palette("colorblind", n_colors=len(topic_sizes))

    # Compute filtered similarity matrix for these topics only
    topic_sim = cosine_similarity(bert_model.topic_embeddings_)
    topic_sim_filtered = topic_sim[np.ix_(topic_ids, topic_ids)]
    np.fill_diagonal(topic_sim_filtered, 0)

    # Build graph from filtered similarity matrix
    adj_matrix = topic_sim_filtered.copy()
    G = ig.Graph.Adjacency((adj_matrix > 0.3).tolist(), mode=ig.ADJ_UNDIRECTED)
    lower_tri = np.tril(adj_matrix, k=-1)
    G.es['weight'] = lower_tri[adj_matrix.nonzero()]

    # Compute layout on filtered embeddings
    t_pos = umap.UMAP(random_state=42).fit_transform(topic_embeddings)
    t_pos = t_pos - t_pos.mean(axis=0)

    # Plot
    fig, ax = plt.subplots(dpi=400, figsize=(10, 10))  
    ig.plot(G,
            target=ax,
            layout=t_pos.tolist(),
            vertex_size=vertex_sizes.tolist(),
            vertex_color=t_colors,
            vertex_label=topic_labs,
            vertex_label_size=5,
            vertex_label_dist=0,
            vertex_frame_width=0,
            edge_width=[2 * w for w in G.es['weight']],
            edge_color='gray',
            edge_arrow_size=0.001,
            edge_curved=0.3)

    ax.set_title("BERTopic Topic Network", fontsize=14)