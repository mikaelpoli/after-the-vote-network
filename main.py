""" SETUP """
# LIBRARIES
import json
import pandas as pd
from pathlib import Path
import praw
import praw.exceptions
import time
import tqdm

# DIRECTORIES
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / 'src'
DATA_DIR = BASE_DIR / 'data'
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR = BASE_DIR / 'results'

# PARAMETERS
keyword = "trans"
top_n = 5

""" AUTHENTICATION """
reddit = praw.Reddit(
    client_id = input("client_id: "),
    client_secret = input("client_secret: "),
    user_agent = input("user_agent: "),
    check_for_async = False
)

""" SELECT SUBREDDITS """
print(f"Searching for subreddits related to the hashtag '{keyword}'...")
related_subreddits = reddit.subreddits.search_by_name(
    keyword,
    include_nsfw=False,
    exact=False
)  

# Get the top 'n' matching subreddits
top_subreddits = list(related_subreddits)[:top_n]
if not top_subreddits:
    print(f"No subreddits related to '{keyword}' found. Exiting.")
    exit()

print(f"\nTop {top_n} subreddits related to the hashtag '{keyword}':")
for i, subreddit in enumerate(top_subreddits):
    print(f"{i + 1}. {subreddit.display_name}")

# Keep the most relevant ones
relevant_subreddits = ['trans', 'transgender']
subreddits = [s for s in top_subreddits if s in relevant_subreddits]

""" GET POSTS """
n_posts = 1000
time_filter = 'year'  

for subreddit in subreddits:
    all_posts = []

    for post in tqdm.tqdm(subreddit.top(limit=n_posts, time_filter=time_filter),
                     total=n_posts, desc=f'Reddit posts from r/{subreddit.display_name}'):
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

    # Save to JSON
    filename = DATA_DIR / f'{subreddit.display_name}_top_posts.json'
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(unique_posts_list, f, ensure_ascii=False, indent=2)