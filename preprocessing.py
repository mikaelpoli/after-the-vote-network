""" DESCRIPTION """
# Clean comments for semantic network anaylsis

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
    print(f"Length df '{key}': {len(comments[key])}")