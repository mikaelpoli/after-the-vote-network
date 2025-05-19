""" SETUP """
# LIBRARIES
from bs4 import BeautifulSoup
import contractions
import emoji
from nltk.tokenize import sent_tokenize
from pathlib import Path
import re
import spacy
from spellchecker import SpellChecker
import sys
import time
from tqdm import tqdm
import unidecode

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


""" FUNCTIONS """
nlp = spacy.load("en_core_web_sm")
spell = SpellChecker()

class CleanText:
    def __init__(self, text, POS_KEEP=["ADJ","ADV","NOUN","PROPN","VERB"], sentence_split=False):
        tic = time.time()
        self.text = list(text)
        self.sentence_split = sentence_split
        sup_clean = [self._superficial_cleaning(i) for i in self.text]
        if sentence_split:
            sup_clean = [sent_tokenize(i) for i in sup_clean]
            sup_clean = [item for sublist in sup_clean for item in sublist]  # Flatten
        self.text_clean = [self._deep_cleaning(i, POS_KEEP) for i in sup_clean]
        self.pos_clean = [self._deep_cleaning_pos(i, POS_KEEP) for i in sup_clean]
        print(f'Cleaning text: execution time {time.time() - tic:.2f} [s]')

    def _superficial_cleaning(self, text):
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text(separator=" ")

        # Remove text inside square brackets
        text = re.sub(r'\[.*?\]', '', text)

        # Remove links
        text = re.sub(r'http\S+|www\S+', '', text)

        # Remove hashtags, mentions
        text = re.sub(r'[@#]\w+', '', text)

        # Remove accented characters
        text = unidecode.unidecode(text)

        # Expand contractions
        text = contractions.fix(text)

        # Remove emoji
        text = emoji.replace_emoji(text, replace='')

        # Remove numbers
        text = re.sub(r'\d+', '', text)

        # Remove moderator messages
        mod_phrases = ['i am a bot', 'automoderator', 'your post has been removed']
        for phrase in mod_phrases:
            if phrase in text.lower():
                return ''

        # Reduce character repetition
        text = re.sub(r'([A-Za-z])\1{2,}', r'\1\1', text)
        text = re.sub(r'([.,/#!$%^&*?;:{}=_`~()])\1{1,}', r'\1', text)

        # Remove punctuation except sentence-ending ones
        text = re.sub(r'[^\w\s\.\!\?]', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Lowercase
        text = text.lower()

        # Spell correction (basic, optional due to cost)
        words = text.split()
        corrected = [spell.correction(w) if w not in spell and len(w) > 3 else w for w in words]
        return ' '.join(corrected)

    def _deep_cleaning(self, text, POS_KEEP):
        return ' '.join([
            token.lemma_ for token in nlp(text)
            if token.pos_ in POS_KEEP and not token.is_stop and token.is_alpha
        ])

    def _deep_cleaning_pos(self, text, POS_KEEP):
        return [' '.join([token.lemma_, token.pos_])
                for token in nlp(text)
                if token.pos_ in POS_KEEP and not token.is_stop and token.is_alpha]


def apply_CleanText(df, col_names, type='posts_title'):
    if type not in ['posts_title', 'posts_body', 'comments']:
        raise ValueError("Type must be 'posts_title', 'posts_body', or 'comments'.")
    
    col = col_names[type]
    
    cleaner = CleanText(df[col].tolist(), sentence_split=False)
    df[f'{col}_clean'] = cleaner.text_clean
    df[f'{col}_clean_pos'] = cleaner.pos_clean

    return df