""" DESCRIPTION """
# Filter top posts by date for selected subreddits
# Download comments from filtered posts


""" SETUP"""
# LIBRARIES
import json
import pandas as pd
from pathlib import Path
import praw
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

# CUSTOM LIBRARIES
import utils


""" IMPORT DATA """
# Load data from JSON
json_files = list(POSTS_ALL_DIR.glob('*.json'))
loaded_data = {}
for file in json_files:
    key = Path(file).stem
    with open(file, 'r', encoding='utf-8') as f:
        loaded_data[key] = json.load(f)

# Convert to df
posts = {}
for key, value in loaded_data.items():
    if isinstance(value, list):
        posts[key] = pd.DataFrame(value)
    print(f"Length df '{key}': {len(posts[key])}")

# Select only posts from December 6, 2024 on
start_date = pd.to_datetime("2024-12-06")
end_date = pd.to_datetime("2025-05-17")
top_posts = utils.filter_posts_by_date(start_date, end_date, posts, to_json=True, dir=POSTS_FILTERED_DIR)


""" FETCH COMMENTS """
# AUTHENTICATION
reddit = praw.Reddit(
    client_id = input("client_id: "),
    client_secret = input("client_secret: "),
    user_agent = input("user_agent: "),
    check_for_async = False
)

# GET COMMENTS
comments_dict = utils.fetch_comments_from_subreddit(reddit, top_posts, to_json=True, dir=COMMENTS_DIR)