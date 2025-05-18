""" DESCRIPTION """
# Download Top posts for selected subreddits Reddit's API

""" SETUP """
# LIBRARIES
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

# PARAMETERS
keyword = "trans"
top_n = 5

""" AUTHENTICATION """
reddit = praw.Reddit(
    client_id = input("client_id: "),
    client_secret = input("client_secret: "),
    user_agent = input("user_agent: "),
    check_for_async = False
)

""" SELECT SUBREDDITS """
related_subreddits = list(reddit.subreddits.search(keyword))
filtered_subreddits = [
    sub for sub in related_subreddits
    if keyword in sub.display_name
    and sub.over18 == False
    and sub.subscribers is not None
]

# Get the top 'n' matching subreddits
top_subreddits = filtered_subreddits[:top_n]
print(f"\nTop {top_n} subreddits for keyword '{keyword}':")
for i, sub in enumerate(top_subreddits):
    print(f"{i + 1}. {sub.display_name}")
print('\n')

# Keep the most relevant ones
relevant_subreddits = ['trans', 'asktransgender', 'transgender']
subreddits = [s for s in top_subreddits if s in relevant_subreddits]

""" FETCH POSTS """
n_posts = 1000
time_filter = 'year'  

retrieved_posts = utils.fetch_posts_from_subreddits(subreddits, n_posts, time_filter, to_json=False, dir=POSTS_ALL_DIR)