""" DESCRIPTION """
# Clean posts and comments for semantic network anaylsis

""" SETUP """
# LIBRARIES
import json
import pandas as pd
from pathlib import Path
import praw
import praw.exceptions
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

# CUSTOM LIBRARIES
import utils

""" IMPORT DATA """
# POSTS
# Load data from JSON
json_files = list(POSTS_FILTERED_DIR.glob('*.json'))
loaded_data = {}
for file in json_files:
    key = Path(file).stem.replace('_filtered', '')
    with open(file, 'r', encoding='utf-8') as f:
        loaded_data[key] = json.load(f)

# Convert to df
posts = {}
for key, value in loaded_data.items():
    if isinstance(value, list):
        posts[key] = pd.DataFrame(value)
    print(f"Length posts df '{key}': {len(posts[key])}")

# COMMENTS
# Load data from JSON
json_files = list(COMMENTS_DIR.glob('*.json'))
loaded_data = {}
for file in json_files:
    key = Path(file).stem.replace('_comments', '')
    with open(file, 'r', encoding='utf-8') as f:
        loaded_data[key] = json.load(f)

# Convert to df
comments = {}
for key, value in loaded_data.items():
    if isinstance(value, list):
        comments[key] = pd.DataFrame(value)
        comments[key]['subreddit'] = key
    print(f"Length comments df '{key}': {len(comments[key])}")