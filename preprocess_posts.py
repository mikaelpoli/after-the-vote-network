""" DESCRIPTION """
# Clean posts and comments for semantic network anaylsis

""" SETUP """
# LIBRARIES
import json
import os
import pandas as pd
from pathlib import Path
import subprocess
import sys
import uuid

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

# UTILITY FUNCTION TO RUN SPACY IN PYTHON 3.11
def clean_batch_with_spacy(text_list):
    # Save input data
    input_file = f"tmp_input_{uuid.uuid4().hex}.json"
    output_file = f"tmp_output_{uuid.uuid4().hex}.json"

    with open(input_file, "w", encoding="utf-8") as f:
        json.dump(text_list, f, ensure_ascii=False, indent=2)

    # Call spaCy cleaner using Python 3.11 environment
    script_path = os.path.join("src", "preprocess.py")
    result = subprocess.run(
        ["spacy-env\\Scripts\\python.exe", script_path, input_file, output_file],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("Error:", result.stderr)
        return None

    # Read and return results
    with open(output_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Clean up temp files
    os.remove(input_file)
    os.remove(output_file)

    return data


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


""" PREPROCESS DATA """
# POSTS
# Titles
all_titles, df_keys_titles, title_indices = utils.extract_relevant_text(posts, col="title")
cleaned_titles = clean_batch_with_spacy(all_titles)
if cleaned_titles is None:
    raise RuntimeError("spaCy cleaning for titles failed.")

preprocessed_posts = utils.assign_cleaned_text_to_dfs(cleaned_titles,
                                                      df_keys_titles,
                                                      title_indices,
                                                      posts,
                                                      clean_col='clean_title',
                                                      clean_pos_col='clean_title_pos')

"""
# Comments
all_comments, df_keys_comments, comment_indices = utils.extract_relevant_text(comments, col="comment_body")
cleaned_comments = clean_batch_with_spacy(all_comments)
if cleaned_comments is None:
    raise RuntimeError("spaCy cleaning for comments failed.")

preprocessed_comments = utils.assign_cleaned_text_to_dfs(cleaned_comments,
                                                         df_keys_comments,
                                                         comment_indices,
                                                         comments,
                                                         clean_col='clean_comment_body',
                                                         clean_pos_col='clean_comment_body_pos')
"""