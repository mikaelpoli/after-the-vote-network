""" SETUP """
# LIBRARIES
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pickle
import powerlaw
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
        self.Mwd = None  # Raw count matrix M(wd)
        self.Pwd = None  # Joint probability P(wd)
        self.Pw_d = None  # P(w|d)
        self.pw = None  # p(w)
        self.pd = None  # p(d)
        self.Pww = None # Projection on words P(ww)
        self.Pdd = None # Projection on documents P(dd)

    def _build_vocab_and_counts(self):
        vocab_set = set()
        doc_word_counts = []

        for _, row in self.df.iterrows():
            entries = row[self.column]
            if not entries:
                entries = []
            words = [token for token in entries if isinstance(token, str)]  # Keep POS intact
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

        # Build Mwd (raw count matrix)
        self.Mwd = np.zeros((num_words, num_docs), dtype=float)
        for j, word_count in enumerate(doc_word_counts):
            for word, count in word_count.items():
                i = self.word_index[word]
                self.Mwd[i, j] = count

        # Compute P(w|d)
        if tfidf:
            tfidf_transformer = TfidfTransformer(norm=None, use_idf=True, smooth_idf=True)
            self.Pw_d = tfidf_transformer.fit_transform(self.Mwd).toarray()
        else:
            col_sums = self.Mwd.sum(axis=0)
            self.Pw_d = np.divide(self.Mwd, col_sums, where=col_sums != 0)

        # Compute P(d) – uniform if not specified
        self.pd = np.ones(num_docs) / num_docs

        # Compute joint probability P(w,d) = P(w|d) * P(d)
        self.Pwd = self.Pw_d * self.pd  # broadcasting P(d) over columns

        # Compute marginal probabilities
        self.pw = self.Pwd @ np.ones(num_docs)
        self.pd = self.Pwd.T @ np.ones(num_words)

        # Compute projections
        with np.errstate(divide='ignore', invalid='ignore'):
            prob_d_inv = np.diag(np.where(self.pd != 0, 1.0 / self.pd, 0.0))
            prob_w_inv = np.diag(np.where(self.pw != 0, 1.0 / self.pw, 0.0))

        self.Pww = self.Pwd @ prob_d_inv @ self.Pwd.T
        self.Pdd = self.Pwd.T @ prob_w_inv @ self.Pwd

    def plot_degree_distribution(self, type='words'):
        if self.Mwd is None:
            raise ValueError("Mwd matrix not built yet. Run the build() method first.")

        if type == 'words':
            degrees = np.squeeze(np.asarray(self.Mwd.sum(axis=1)))  # sum over documents
            title = "Degree Distribution for Words"
        elif type == 'documents':
            degrees = np.squeeze(np.asarray(self.Mwd.sum(axis=0)))  # sum over words
            title = "Degree Distribution for Documents"
        else:
            raise ValueError("type must be either 'words' or 'documents'")

        # Compute degree histogram
        k = np.unique(degrees)
        pk = np.histogram(degrees, bins=np.append(k, k[-1] + 1))[0]
        pk = pk / pk.sum()  # normalize

        # Plot on log-log scale
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

        # Get degrees
        if type == 'words':
            degrees = np.squeeze(np.asarray(self.Mwd.sum(axis=1)))  # word degrees
        elif type == 'documents':
            degrees = np.squeeze(np.asarray(self.Mwd.sum(axis=0)))  # doc degrees
        else:
            raise ValueError("type must be 'words' or 'documents'")

        # Remove zeros
        degrees = degrees[degrees > 0]

        # Fit power-law model
        fit = powerlaw.Fit(degrees, discrete=True, verbose=False)

        # Print summary
        print(f"--- Power-Law Analysis ({type}) ---")
        print(f"Alpha (scaling exponent): {fit.power_law.alpha:.4f}")
        print(f"xmin (cutoff): {fit.power_law.xmin}")

        # Plot
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


def to_networkx_bipartite(builder, use='Pwd'):
    if use not in ['Pwd', 'Mwd']:
        raise ValueError("'use' must be one of 'Pwd' or 'Mwd'")
    
    B = nx.Graph()
    words = builder.words
    documents = builder.documents
    matrix = getattr(builder, use)

    num_words, num_docs = matrix.shape

    # Add nodes with bipartite attribute
    B.add_nodes_from(words, bipartite='word')
    B.add_nodes_from(documents, bipartite='document')

    # Add edges
    for i, word in enumerate(words):
        for j, doc in enumerate(documents):
            weight = matrix[i, j]
            if weight > 0:
                B.add_edge(word, doc, weight=weight)

    return B


def to_networkx_word_projection(builder, use='Pww', threshold=0.0):
    if use not in ['Pww']:
        raise ValueError("'use' must be 'Pww'")
    
    G = nx.Graph()
    words = builder.words
    matrix = getattr(builder, use)
    num_words = matrix.shape[0]

    # Add word nodes
    G.add_nodes_from(words)

    # Add edges (symmetric matrix)
    for i in range(num_words):
        for j in range(i + 1, num_words):  # Avoid duplicates
            weight = matrix[i, j]
            if weight > threshold:
                word_i = words[i]
                word_j = words[j]
                G.add_edge(word_i, word_j, weight=weight)

    return G