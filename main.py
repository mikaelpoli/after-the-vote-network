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
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / 'src'
sys.path.append(str(SRC_DIR))
DATA_DIR = BASE_DIR / 'data'
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR = BASE_DIR / 'results'

""" IMPORT DATA """
# Load data from JSON
json_files = list(DATA_DIR.glob('*.json'))
loaded_data = {}
for file in json_files:
    key = Path(file).stem
    with open(file, 'r', encoding='utf-8') as f:
        loaded_data[key] = json.load(f)

# Convert to df
dataframes = {}
for key, value in loaded_data.items():
    if isinstance(value, list):
        dataframes[key] = pd.DataFrame(value)
    print(f"Length df '{key}': {len(dataframes[key])}")