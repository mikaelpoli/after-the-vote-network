""" DESCRIPTION """
# Build the semantic network from clean Reddit posts' comments for network anaylsis


""" SETUP """
# LIBRARIES
import networkx as nx
import pandas as pd
from pathlib import Path
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
POSTS_FILTERED_CLEAN_DIR = POSTS_FILTERED_DIR / 'clean'
COMMENTS_DIR = DATA_DIR / 'comments'
COMMENTS_CLEAN_DIR = COMMENTS_DIR / 'clean'
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_GRAPHS_DIR = RESULTS_DIR / 'graphs'

# CUSTOM LIBRARIES
import build_network as bn


""" IMPORT DATA """
# COMMENTS
# Load data from JSON
filename = COMMENTS_CLEAN_DIR / 'all_comments_clean.json'
comments = pd.read_json(filename)


""" BUILD BIPARTITE NETWORK """
network = bn.BuildNetwork(comments)
network.build(tfidf=False)

# Build bipartite graph
G = bn.to_networkx_bipartite(network, use='Pwd')

# Visualize or analyze
print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# Save to Gephi-compatible format
filename = RESULTS_GRAPHS_DIR / 'comments.gexf'
nx.write_gexf(G, filename)

""" ANALYZE DEGREE DISTRIBUTION """
network.plot_degree_distribution(type='words')
network.plot_degree_distribution(type='documents')
network.analyze_degree_distribution_powerlaw(type='words')
network.analyze_degree_distribution_powerlaw(type='documents')