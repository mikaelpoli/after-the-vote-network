""" SETUP """
# LIBRARIES
import json
from pathlib import Path
import tqdm

# DIRECTORIES
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / 'src'
DATA_DIR = BASE_DIR / 'data'
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR = BASE_DIR / 'results'


""" FUNCTIONS """
def fetch_posts_from_subreddits(subreddits, n_posts, time_filter, to_json=True):
    retrieved_posts = {}
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
        retrieved_posts[subreddit.display_name] = unique_posts_list
        
        # Save to JSON
        if to_json:
            filename = DATA_DIR / f'{subreddit.display_name}_top_posts.json'
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(unique_posts_list, f, ensure_ascii=False, indent=2)

    return retrieved_posts