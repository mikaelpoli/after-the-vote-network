""" SETUP """
# LIBRARIES
from collections import defaultdict
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd
import pickle
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
    def __init__(self, df: pd.DataFrame, column: str = "filtered_pos"):
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

    def build(self, tfidf: bool = False):
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

    def pickle_export(self, filename: str):
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

    def pickle_import(self, filename: str):
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