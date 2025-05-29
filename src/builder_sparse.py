""" SETUP """
# LIBRARIES
from collections import defaultdict
from pathlib import Path
import igraph as ig
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pickle
import powerlaw
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
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
class BuildNetwork:
    def __init__(self, df, column="filtered_pos"):
        self.df = df
        self.column = column
        self.words = []
        self.documents = df.index.tolist()
        self.word_index = {}
        self.Mwd = None  # Sparse matrix (words x documents)
        self.Pwd = None  # Sparse joint probability matrix
        self.Pw_d = None  # Sparse conditional probability matrix
        self.pw = None  # marginal probability vector (words)
        self.pd = None  # marginal probability vector (documents)
        self.Pww = None # Sparse projection on words
        self.Pdd = None # Sparse projection on documents

    def _build_vocab_and_counts(self):
        vocab_set = set()
        doc_word_counts = []

        for _, row in self.df.iterrows():
            entries = row[self.column]
            if not entries:
                entries = []
            words = [token for token in entries if isinstance(token, str)]
            word_count = defaultdict(int)
            for word in words:
                vocab_set.add(word)
                word_count[word] += 1
            doc_word_counts.append(word_count)

        self.words = sorted(vocab_set)
        self.word_index = {word: i for i, word in enumerate(self.words)}
        return doc_word_counts

    def build(self, tfidf=False):
        doc_word_counts = self._build_vocab_and_counts()
        num_words = len(self.words)
        num_docs = len(self.documents)

        # Prepare data for sparse matrix construction
        data = []
        row_indices = []
        col_indices = []
        for j, word_count in enumerate(doc_word_counts):
            for word, count in word_count.items():
                i = self.word_index[word]
                row_indices.append(i)
                col_indices.append(j)
                data.append(count)

        # Build sparse raw count matrix (csr_matrix)
        self.Mwd = sp.csr_matrix((data, (row_indices, col_indices)), shape=(num_words, num_docs), dtype=float)

        if tfidf:
            tfidf_transformer = TfidfTransformer(norm=None, use_idf=True, smooth_idf=True)
            self.Pw_d = tfidf_transformer.fit_transform(self.Mwd)
        else:
            # Column sums (document sums)
            col_sums = np.array(self.Mwd.sum(axis=0)).flatten()
            # Create diagonal matrix of 1/col_sums safely
            inv_col_sums = np.divide(1.0, col_sums, out=np.zeros_like(col_sums), where=col_sums!=0)
            D_inv = sp.diags(inv_col_sums)
            self.Pw_d = self.Mwd @ D_inv  # Each column sums to 1

        # Uniform document probabilities
        self.pd = np.ones(num_docs) / num_docs

        # Compute joint probability P(w,d) = P(w|d) * P(d)
        # P(w|d) is sparse; multiply each column by pd[j]
        pd_diag = sp.diags(self.pd)
        self.Pwd = self.Pw_d @ pd_diag  # shape (num_words, num_docs)

        # Marginal probabilities
        self.pw = np.array(self.Pwd.sum(axis=1)).flatten()
        self.pd = np.array(self.Pwd.sum(axis=0)).flatten()

        # Inverse probability diagonal matrices for projection
        inv_pd = np.divide(1.0, self.pd, out=np.zeros_like(self.pd), where=self.pd != 0)
        inv_pw = np.divide(1.0, self.pw, out=np.zeros_like(self.pw), where=self.pw != 0)

        D_pd_inv = sp.diags(inv_pd)
        D_pw_inv = sp.diags(inv_pw)

        # Compute projections:
        self.Pww = self.Pwd @ D_pd_inv @ self.Pwd.T
        self.Pdd = self.Pwd.T @ D_pw_inv @ self.Pwd

    def plot_degree_distribution(self, type='words'):
        if self.Mwd is None:
            raise ValueError("Mwd matrix not built yet. Run the build() method first.")

        if type == 'words':
            degrees = np.array(self.Mwd.sum(axis=1)).flatten()
            title = "Degree Distribution for Words"
        elif type == 'documents':
            degrees = np.array(self.Mwd.sum(axis=0)).flatten()
            title = "Degree Distribution for Documents"
        else:
            raise ValueError("type must be either 'words' or 'documents'")

        k = np.unique(degrees)
        pk, _ = np.histogram(degrees, bins=np.append(k, k[-1] + 1))
        pk = pk / pk.sum()

        plt.figure(figsize=(4, 3))
        plt.loglog(k, pk, 'o', markersize=5)
        plt.title(title)
        plt.xlabel("k (degree)")
        plt.ylabel("p(k)")
        plt.grid(True, which="both", ls="--", lw=0.5)
        plt.tight_layout()
        plt.show()

    def analyze_degree_distribution_powerlaw(self, type='words', plot=True):
        if self.Mwd is None:
            raise ValueError("Mwd matrix not built yet. Call build() first.")

        if type == 'words':
            degrees = np.array(self.Mwd.sum(axis=1)).flatten()
        elif type == 'documents':
            degrees = np.array(self.Mwd.sum(axis=0)).flatten()
        else:
            raise ValueError("type must be 'words' or 'documents'")

        degrees = degrees[degrees > 0]
        fit = powerlaw.Fit(degrees, discrete=True, verbose=False)

        print(f"--- Power-Law Analysis ({type}) ---")
        print(f"Gamma (scaling exponent): {fit.power_law.alpha:.4f}")
        print(f"K_min (cutoff): {fit.power_law.xmin}")

        if plot:
            fig = fit.plot_pdf(label='Empirical', color='blue')
            fit.power_law.plot_pdf(label='Power law fit', color='red', ax=fig)
            plt.title(f"Power-Law Fit ({type})")
            plt.xlabel("Degree (k)")
            plt.ylabel("p(k)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def pickle_export(self, filename):
        out_data = {
            'words': self.words,
            'documents': self.documents,
            'Mwd': self.Mwd,
            'Pwd': self.Pwd,
            'Pww': self.Pww,
            'Pdd': self.Pdd,
            'pd': self.pd,
            'pw': self.pw,
        }
        with open(filename, 'wb') as f:
            pickle.dump(out_data, f)

    def pickle_import(self, filename):
        with open(filename, 'rb') as f:
            in_data = pickle.load(f)
        self.words = in_data['words']
        self.documents = in_data['documents']
        self.Mwd = in_data['Mwd']
        self.Pwd = in_data['Pwd']
        self.Pww = in_data['Pww']
        self.Pdd = in_data['Pdd']
        self.pd = in_data['pd']
        self.pw = in_data['pw']


def to_igraph_bipartite(builder, use='Pwd'):
    if use not in ['Pwd', 'Mwd']:
        raise ValueError("'use' must be one of 'Pwd' or 'Mwd'")

    words = builder.words
    documents = builder.documents
    matrix = getattr(builder, use)

    num_words, num_docs = matrix.shape
    vertices = words + documents
    bipartite_attr = ['word'] * num_words + ['document'] * num_docs

    edges = []
    edge_weights = []

    # matrix is sparse; iterate nonzero entries efficiently
    matrix_coo = matrix.tocoo()
    for i, j, weight in zip(matrix_coo.row, matrix_coo.col, matrix_coo.data):
        if weight > 0:
            edges.append((i, num_words + j))
            edge_weights.append(weight)

    g = ig.Graph(edges=edges)
    g.add_vertices(len(vertices) - len(g.vs))
    g.vs['name'] = vertices
    g.vs['bipartite'] = bipartite_attr
    g.es['weight'] = edge_weights

    return g


def to_igraph_projected(builder, use='Pww', threshold=0.0):
    if use not in ['Pww', 'Pdd']:
        raise ValueError("'use' must be 'Pww' or 'Pdd'")

    if use == 'Pww':
        nodes = builder.words
    else:
        nodes = builder.documents

    matrix = getattr(builder, use)

    # Convert to COO format for easy data access
    matrix_coo = matrix.tocoo()

    # Extract upper triangle indices (i < j) and their weights
    upper_mask = matrix_coo.row < matrix_coo.col
    row_upper = matrix_coo.row[upper_mask]
    col_upper = matrix_coo.col[upper_mask]
    data_upper = matrix_coo.data[upper_mask]

    if threshold > 0:
        # Compute the actual weight cutoff value at the given percentile
        cutoff = np.percentile(data_upper, threshold * 100)
        # Keep edges with weight > cutoff
        mask = data_upper > cutoff
        row_upper = row_upper[mask]
        col_upper = col_upper[mask]
        data_upper = data_upper[mask]
    else:
        # If threshold=0, keep all edges in upper triangle
        pass

    edges = list(zip(row_upper, col_upper))
    edge_weights = data_upper.tolist()

    g = ig.Graph(edges=edges)
    g.add_vertices(len(nodes) - len(g.vs))
    g.vs['name'] = nodes
    g.es['weight'] = edge_weights

    return g


def filter_sparse_by_percentile(matrix, percentile):
    coo = matrix.tocoo()
    cutoff = np.percentile(coo.data, percentile)
    
    mask = coo.data > cutoff
    
    rows = coo.row[mask]
    cols = coo.col[mask]
    data = coo.data[mask]
    
    filtered_matrix = sp.coo_matrix((data, (rows, cols)), shape=matrix.shape)
    
    return filtered_matrix.tocsr()