""" SETUP """
# LIBRARIES
import pandas as pd
import praw
import praw.exceptions
import time
import tqdm

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

""" SEARCH SUBREDDITS """
print(f"Searching for subreddits related to the hashtag '{keyword}'...")
related_subreddits = reddit.subreddits.search_by_name(
    keyword,
    include_nsfw=False,
    exact=False
)  

# Get the top 'n' matching subreddits
top_subreddits = list(related_subreddits)[:top_n]
if not top_subreddits:
    print(f"No subreddits related to '{keyword}' found. Exiting.")
    exit()

print(f"\nTop {top_n} subreddits related to the hashtag '{keyword}':")
for i, subreddit in enumerate(top_subreddits):
    print(f"{i + 1}. {subreddit.display_name}")