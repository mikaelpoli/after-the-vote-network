""" SETUP """
# LIBRARIES
from bs4 import BeautifulSoup
from IPython.display import HTML
from pathlib import Path
from scipy.sparse import csr_matrix
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