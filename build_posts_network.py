""" DESCRIPTION """
# Build the semantic network from clean Reddit posts for network anaylsis


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
import utils
import preprocess as prep
import build_network as bn


""" IMPORT DATA """
# POSTS
# Load data from JSON
filename = POSTS_FILTERED_CLEAN_DIR / 'all_posts_clean.json'
posts = pd.read_json(filename)

print(posts)


""" BUILD BIPARTITE NETWORK """
builder = bn.BuildNetwork(posts)
builder.build(tfidf=False)

# Build bipartite graph
G = bn.to_networkx_bipartite(builder, use='Pwd')

# Visualize or analyze
print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# Save to Gephi-compatible format
filename = RESULTS_GRAPHS_DIR / 'posts.gexf'
nx.write_gexf(G, filename)