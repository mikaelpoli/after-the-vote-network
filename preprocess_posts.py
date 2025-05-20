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
                                     to_json=False)

# Selftext (body)
posts_selftext_clean = prep.clean_data(posts_titles_clean, 
                                       col_names,
                                       clean_type='posts_body',
                                       sentence_split=True,
                                       use_spellcheck=True,
                                       to_json=False)


# KEEP RELEVANT COLUMNS
columns_to_keep = [
    "subreddit",
    "upvote_ratio",
    "ups",
    "created_utc",
    "num_comments",
    "id",
    "title_clean",
    "title_clean_pos",
    "selftext_clean",
    "selftext_clean_pos"
]

# Filter each dataframe in the dictionary
for key in posts_selftext_clean:
    posts_selftext_clean[key] = posts_selftext_clean[key][columns_to_keep]

# Save to JSON
for key, df in posts_selftext_clean.items():
    filename = POSTS_FILTERED_CLEAN_DIR / f'{key}_clean.json'
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(df.to_dict(orient='records'), f, ensure_ascii=False, indent=2)
    print(f"Saved {len(df)} comments from r/{key} to JSON")