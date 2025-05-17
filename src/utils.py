""" SETUP """
# LIBRARIES
import json
import pandas as pd
from pathlib import Path
import sys
import tqdm

# DIRECTORIES
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / 'src'
sys.path.append(str(SRC_DIR))
DATA_DIR = BASE_DIR / 'data'
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR = BASE_DIR / 'results'


""" FUNCTIONS """
def fetch_posts_from_subreddits(subreddits, n_posts, time_filter, to_json=True):
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
            filename = DATA_DIR / f'{key}.json'
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(unique_posts_list, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(unique_posts_list)} posts from r/{key} to JSON")
        
    return retrieved_posts

def filter_posts_by_date(start_date, end_date, dataframes):
    for key, df, in dataframes.items():
        df['created_datetime'] = pd.to_datetime(df['created_utc'], unit='s')
        filtered_df = df[(df['created_datetime'] >= start_date) & (df['created_datetime'] <= end_date)]
        dataframes[key] = filtered_df
        print(f"After filtering '{key}': {len(filtered_df)} posts")
    
    print("Filtered dataframes by date.")
    return dataframes