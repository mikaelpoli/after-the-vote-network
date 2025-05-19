""" DESCRIPTION """
# Clean posts' title and selftext (body) for semantic network anaylsis


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


""" PREPROCESS DATA """
# Get names of columns where relevant text is stored
col_names = {
    'posts_title': "title",
    'posts_body': "selftext",
}

# POSTS
# Titles
posts_titles_clean = prep.clean_data(posts, 
                                     col_names,
                                     clean_type='posts_title',
                                     sentence_split=False,
                                     use_spellcheck=True,
                                     to_json=True, dir=POSTS_FILTERED_CLEAN_DIR)

# Selftext (body)
posts_selftext_clean = prep.clean_data(posts_titles_clean, 
                                       col_names,
                                       clean_type='posts_body',
                                       sentence_split=True,
                                       use_spellcheck=True,
                                       to_json=True, dir=POSTS_FILTERED_CLEAN_DIR)