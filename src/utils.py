""" SETUP """
# LIBRARIES
import copy
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import linregress
import sys
import time
import tqdm

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
def fetch_posts_from_subreddits(subreddits, n_posts, time_filter, to_json=True, dir=None):
    retrieved_posts = {}
    for subreddit in subreddits:
        key = subreddit.display_name
        all_posts = []

        for post in tqdm.tqdm(subreddit.top(limit=n_posts, time_filter=time_filter),
                        total=n_posts, desc=f'Reddit posts from r/{key}'):
            all_posts.append({
                'subreddit': post.subreddit.display_name,
                'selftext': post.selftext,
                'author_fullname': post.author_fullname if post.author else 'N/A',
                'title': post.title,
                'upvote_ratio': post.upvote_ratio,
                'ups': post.ups,
                'created': post.created,
                'created_utc': post.created_utc,
                'num_comments': post.num_comments,
                'author': str(post.author) if post.author else 'N/A',
                'id': post.id
            })

        # Remove duplicates by post ID
        unique_posts = {post['id']: post for post in all_posts}
        unique_posts_list = list(unique_posts.values())
        retrieved_posts[key] = unique_posts_list
        print(f"Retrieved {len(retrieved_posts[key])} posts from r/{key}")
        
        # Save to JSON
        if to_json:
            filename = dir / f'{key}.json'
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(unique_posts_list, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(unique_posts_list)} posts from r/{key} to JSON")
        
    return retrieved_posts


def filter_posts_by_date(start_date, end_date, dataframes, to_json=True, dir=None):
    dfs = copy.deepcopy(dataframes)
    for key, df, in dfs.items():
        df['created_datetime'] = pd.to_datetime(df['created_utc'], unit='s')
        filtered_df = df[(df['created_datetime'] >= start_date) & (df['created_datetime'] <= end_date)]
        filtered_df['created_datetime'] = filtered_df['created_datetime'].astype(str)
        dfs[key] = filtered_df
        print(f"After filtering '{key}': {len(filtered_df)} posts")

        if to_json:
            filename = dir / f'{key}_filtered.json'
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(filtered_df.to_dict(orient='records'), f, ensure_ascii=False, indent=2)
            print(f"Saved {len(filtered_df)} filtered posts from r/{key} to JSON")

    print("Filtered dataframes by date.")
    return dfs


def fetch_comments_from_subreddit(api, posts_df, to_json=True, dir=None):
    comments = {}
    for key, df in posts_df.items():
        comments_list = []

        for i in tqdm.tqdm(range(len(df)), desc=f'Reddit comments for r/{key}\'s top posts'):
            post_id = df.iloc[i]['id']
            submission = api.submission(id=post_id)
            try:
                submission.comments.replace_more(limit=None)
                for comment in submission.comments.list():
                    comments_list.append({
                            'comment_id': comment.id,
                            'parent_id': comment.parent_id,
                            'post_id': post_id,
                            'comment_body': comment.body
                        })

            except Exception as e:
                print(f"Error processing post {post_id}: {e}")
                time.sleep(3)

        # Remove duplicates by post ID
        unique_comments = {c['comment_id']: c for c in comments_list}
        unique_comments_list = list(unique_comments.values())
        comments[key] = unique_comments_list
        print(f"Retrieved {len(comments[key])} comments from r/{key}")

        # Save to JSON
        if to_json:
            filename = dir / f'{key}_comments.json'
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(unique_comments_list, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(unique_comments_list)} comments from r/{key} to JSON")
    
    return comments


def plot_giant_component_degree_distribution(graph):
    if graph is None or graph.vcount() == 0:
        raise ValueError("Input graph is empty or None.")

    # Compute degrees
    degrees = np.array(graph.degree())

    # Degree histogram (normalized)
    k = np.unique(degrees)
    pk = np.histogram(degrees, bins=np.append(k, k[-1] + 1))[0]
    pk = pk / pk.sum()

    # Plot on log-log scale
    plt.figure(figsize=(4, 3))
    plt.loglog(k, pk, 'o', markersize=5)
    plt.title("Degree Distribution (Giant Component)")
    plt.xlabel("k (degree)")
    plt.ylabel("p(k)")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.show()


def plot_ccdf(graph):
    degrees = graph.degree()
    degree_values = np.sort(np.unique(degrees))
    ccdf = [np.sum(np.array(degrees) >= k) / len(degrees) for k in degree_values]
    ccdf = np.array(ccdf)

    # Only fit the tail (degrees > 10)
    mask = degree_values > 10
    log_x = np.log10(degree_values[mask])
    log_y = np.log10(ccdf[mask])
    slope, intercept, _ , _ , _ = linregress(log_x, log_y)

    # Plot the fit line
    plt.figure(figsize=(6,4))
    plt.loglog(degree_values, ccdf, marker='o', linestyle='none', label='CCDF')
    plt.loglog(degree_values[mask], 10**(intercept + slope * log_x), linestyle='--', color='orange', label=f'Fit slope={slope:.2f}')
    plt.xlabel('Degree')
    plt.ylabel('CCDF')
    plt.title('CCDF with Power-Law Fit')
    plt.legend()
    plt.show()


