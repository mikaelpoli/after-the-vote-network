""" SETUP """
# LIBRARIES
from collections import defaultdict
import copy
from pathlib import Path
from scipy.sparse import csr_matrix, find, lil_matrix
from sklearn.metrics.pairwise import cosine_similarity
import igraph as ig
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sps
import seaborn as sns
import sys
import umap
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="'force_all_finite' was renamed")
warnings.filterwarnings("ignore", category=UserWarning, message="n_neighbors is larger than the dataset size")


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


def get_louvain_community_reps(g_dd, topics, top_k=10):
    # Ensure edge weights are numeric
    if "weight" in g_dd.es.attribute_names():
        g_dd.es["weight"] = list(map(float, g_dd.es["weight"]))

    # Build vertex-topic lookup
    vertex_topic = np.array(topics)

    # Compute top-K representatives
    community_reps = defaultdict(list)
    unique_topics = np.unique(vertex_topic)

    for t_id in unique_topics:
        v_idx = np.where(vertex_topic == t_id)[0].tolist()
        if not v_idx:
            continue

        sub = g_dd.subgraph(v_idx)
        strengths = sub.strength(weights=sub.es["weight"]) if sub.ecount() else [0] * sub.vcount()

        top_local = np.argsort(strengths)[::-1][:min(top_k, len(strengths))]
        top_global = [v_idx[i] for i in top_local]
        top_docids = [g_dd.vs[g_idx]["name"] for g_idx in top_global]

        community_reps[t_id] = top_docids

    return community_reps


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


def print_top_docs(topic_number, representative_docs_dict):
    print(f"Topic {topic_number}\n")
    counter = 1
    for v in representative_docs_dict[topic_number]:
        print(f"\"Document {counter}: {v}\"\n")
        counter += 1


def plot_topic_pattern(num_docs, C, title_1='BERTopic Relative cluster sizes', title_2='BERTopic Cluster block pattern'):
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
    ax[0].set_title(title_1)
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
    ax[1].set_title(title_2)

    plt.tight_layout()
    plt.show()


def plot_topic_network(bert_model, topic_labels_dict, vertex_size=0.6, vertex_lab_size=5, title="BERTopic Topic Network"):
    # Filter out -1 topic and get consistent topic ids
    topic_info = bert_model.get_topic_info().sort_values("Topic")
    topic_info = topic_info[topic_info["Topic"] != -1]
    topic_ids = topic_info["Topic"].tolist()

    # Get embeddings and other info in correct order
    topic_embeddings = np.array([bert_model.topic_embeddings_[tid] for tid in topic_ids])
    topic_sizes = topic_info["Count"].values
    topic_labs = [topic_labels_dict[tid] for tid in topic_ids]
    vertex_sizes = vertex_size * topic_sizes
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
            vertex_label_size=vertex_lab_size,
            vertex_label_dist=0,
            vertex_frame_width=0,
            edge_width=[2 * w for w in G.es['weight']],
            edge_color='gray',
            edge_arrow_size=0.001,
            edge_curved=0.3)

    ax.set_title(title, fontsize=14)


def plot_topic_network_louvain(Pcc, pc, topic_labels=None, vertex_lab_size=5, title="Topic Graph"):
    # UMAP layout
    t_pos = umap.UMAP().fit_transform(Pcc.toarray())
    t_pos -= t_pos.mean(axis=0)

    # Metadata
    topic_centrality = np.array(pc)[0]
    if topic_labels is not None:
        topic_names = [topic_labels[i] for i in topic_labels]
    else:
        topic_names = [f"topic {i}" for i in range(pc.shape[1])]
    topic_colors = sns.color_palette("colorblind", n_colors=len(topic_names))

    # Build graph
    A = Pcc.toarray()
    np.fill_diagonal(A, 0)
    G = ig.Graph.Adjacency((A > 0).tolist())

    # Use lower triangle of A for edge weights (assumes symmetric matrix)
    At = np.tril(A, k=0)
    G.es["weight"] = np.array(At[A.nonzero()])

    # Plot graph
    fig, ax = plt.subplots(dpi=400, figsize=(10, 10))
    ig.plot(
        G,
        target=ax,
        layout=t_pos,
        vertex_size=1500 * topic_centrality,
        vertex_color=topic_colors,
        vertex_label=topic_names,
        vertex_label_size=vertex_lab_size,
        vertex_label_dist=0,
        vertex_frame_width=0,
        edge_width=100 * np.array(G.es["weight"]),
        edge_color="grey",
        edge_arrow_size=1e-5
    )
    ax.set_title(title, fontsize=14)


def to_bertopic_style(bert_model_in, documents, topics, fine_tuned_labels=None):
    bert_model = copy.deepcopy(bert_model_in)

    documents = pd.DataFrame(documents, columns=['Document'])
    documents['Topic'] = topics

    bert_model.topics_ = documents['Topic'].tolist()

    documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
    c_tf_idf_, words = bert_model._c_tf_idf(documents_per_topic)
    bert_model.c_tf_idf_ = c_tf_idf_

    topic_representations_ = bert_model._extract_words_per_topic(words, documents)
    bert_model.topic_representations_ = topic_representations_
    if fine_tuned_labels is not None:
        bert_model.set_topic_labels(fine_tuned_labels)
    else:
        bert_model.set_topic_labels([f"{key}_" + "_".join([word[0] for word in values[:4]])]
                                for key, values in bert_model.topic_representations_.items())
    
    return bert_model