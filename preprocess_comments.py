""" DESCRIPTION """
# Clean comments for semantic network anaylsis


""" SETUP """
# LIBRARIES
import json
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

# CUSTOM LIBRARIES
import preprocess as prep

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


""" PREPROCESS DATA """
# Get names of column(s) where relevant text is stored
col_names = {
    'comments': "comment_body"
}

# COMMENTS
comments_clean = prep.clean_data(comments, 
                                 col_names,
                                 clean_type='comments',
                                 sentence_split=True,
                                 use_spellcheck=True,
                                 to_json=True, dir=COMMENTS_CLEAN_DIR)