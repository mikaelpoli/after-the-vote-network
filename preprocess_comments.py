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


""" IMPORT DATA """
# COMMENTS
# Load data from JSON
json_files = list(COMMENTS_CLEAN_DIR.glob('*.json'))
loaded_data = {}
for file in json_files:
    key = Path(file).stem.replace('_clean', '')
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
# Get names of columns where relevant text is stored
col_names = {
    'comments': "comment_body",
}

# COMMENTS
comments_clean_start = prep.clean_data(comments,
                                 col_names,
                                 clean_type='comments',
                                 sentence_split=True,
                                 use_spellcheck=True,
                                 to_json=False)

# Concatenate subreddit dataframes
all_comments_clean = pd.DataFrame()
for key, df in comments_clean_start.items():
    all_comments_clean = pd.concat([all_comments_clean, df], ignore_index=True)

# REMOVE RARE WORDS
word_freq = prep.count_word_frequencies(all_comments_clean, pos_col='comment_body_clean_pos')
prep.plot_word_frequencies(word_freq, title="Word Frequencies Before Filtering")
all_comments_clean_filtered, common_words = prep.filter_rare_words(all_comments_clean, word_freq, pos_col='comment_body_clean_pos')
word_freq_filtered = prep.count_word_frequencies(all_comments_clean_filtered, pos_col='filtered_pos')
prep.plot_word_frequencies(word_freq_filtered, title="Word Frequencies After Filtering")

print(f"N documents: {len(all_comments_clean_filtered)}; N words: {len(common_words)}")
print(f"Total nodes: {len(all_comments_clean_filtered) + len(common_words)}")

# KEEP RELEVANT COLUMNS
columns_to_keep = [
    "subreddit",
    "comment_id",
    "parent_id",
    "post_id",
    "comment_body",
    "filtered_pos"
]

comments_clean = all_comments_clean_filtered[columns_to_keep]

# Save to JSON
filename = COMMENTS_CLEAN_DIR / 'all_comments_clean.json'
with open(filename, 'w', encoding='utf-8') as f:
    json.dump(comments_clean.to_dict(orient='records'), f, ensure_ascii=True, indent=2)
print(f"Saved {len(comments_clean)} comments to JSON")