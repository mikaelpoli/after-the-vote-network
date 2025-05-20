""" SETUP """
# LIBRARIES
from bs4 import BeautifulSoup
import contractions
import emoji
import json
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
# Load NLP model
nlp = spacy.load("en_core_web_sm")  # Switch to "en_core_web_trf" for better performance if GPU is available
spell = SpellChecker()


def safe_correct(w):
    corrected = spell.correction(w)
    return corrected if corrected else w


class CleanText:
    def __init__(self, text, POS_KEEP=["ADJ", "ADV", "PRON", "NOUN", "PROPN", "VERB"],
                 sentence_split=False, use_spellcheck=True):

        tic = time.time()
        self.raw_texts = list(text)
        self.POS_KEEP = POS_KEEP
        self.sentence_split = sentence_split
        self.use_spellcheck = use_spellcheck

        # Step 1: Superficial cleaning (preserve casing for spaCy)
        cleaned_texts = [self._superficial_cleaning(t) for t in self.raw_texts]

        # Step 2: Optional sentence splitting
        if sentence_split:
            cleaned_texts = [sent_tokenize(t) for t in cleaned_texts]
            cleaned_texts = [sent for sublist in cleaned_texts for sent in sublist]

        # Step 3: POS filtering and lemmatization (with original casing)
        self.text_clean = [self._deep_cleaning(t) for t in cleaned_texts]
        self.pos_clean = [self._deep_cleaning_pos(t) for t in cleaned_texts]

        print(f'Cleaning text: execution time {time.time() - tic:.2f} [s]')

    def _superficial_cleaning(self, text):
        # Skip cleaning if the whole text is just a URL
        if re.match(r'^\s*(https?://\S+|www\.\S+)\s*$', text):
            return ''

        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text(separator=" ")

        text = re.sub(r'\[.*?\]', '', text)  # Remove bracketed content
        text = re.sub(r'http\S+|www\S+', '', text)  # Remove links
        text = re.sub(r'[@#]\w+', '', text)  # Remove mentions and hashtags
        text = unidecode.unidecode(text)  # Normalize accented characters
        text = contractions.fix(text)  # Expand contractions
        text = emoji.replace_emoji(text, replace='')  # Remove emojis
        text = re.sub(r'\d+', '', text)  # Remove digits

        # Remove common mod-bot messages
        mod_phrases = ['i am a bot', 'automoderator', 'your post has been removed']
        for phrase in mod_phrases:
            if phrase in text.lower():
                return ''

        # Reduce character repetitions
        text = re.sub(r'([A-Za-z])\1{2,}', r'\1\1', text)
        text = re.sub(r'([.,/#!$%^&*?;:{}=_`~()])\1{1,}', r'\1', text)

        # Remove non-sentence punctuation
        text = re.sub(r'[^\w\s\.\!\?]', '', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text  # Note: Keep original casing here

    def _deep_cleaning(self, text):
        doc = nlp(text)
        words = [
            token.lemma_ for token in doc
            if token.pos_ in self.POS_KEEP and token.is_alpha and (
                token.pos_ != "PRON" or True  # Keep PRONs even if stopwords
            )
        ]

        if self.use_spellcheck:
            words = [
                safe_correct(w) if w not in spell and len(w) > 3 else w
                for w in words
            ]

        return ' '.join(words).lower()

    def _deep_cleaning_pos(self, text):
        doc = nlp(text)
        return [
            f"{token.lemma_.lower()} {token.pos_}"
            for token in doc
            if token.pos_ in self.POS_KEEP and token.is_alpha and (
                token.pos_ != "PRON" or True
            )
        ]
        

def apply_CleanText(df, col_names, type='posts_title', sentence_split=False, use_spellcheck=False):
    if type not in ['posts_title', 'posts_body', 'comments']:
        raise ValueError("Type must be 'posts_title', 'posts_body', or 'comments'.")
    
    col = col_names[type]
    
    cleaner = CleanText(df[col].tolist(), sentence_split=sentence_split, use_spellcheck=use_spellcheck)
    df[f'{col}_clean'] = cleaner.text_clean
    df[f'{col}_clean_pos'] = cleaner.pos_clean

    return df


def clean_data(dataframes_dict, col_names, clean_type='posts_title', sentence_split=False, use_spellcheck=False, to_json=False, dir=None):
    cleaned_data = {}

    for key, df in dataframes_dict.items():
        cleaned_df = apply_CleanText(df, col_names, type=clean_type, sentence_split=False, use_spellcheck=False)
        if cleaned_df is None:
            raise RuntimeError(f"Cleaning for {clean_type} for key '{key}' failed.")
        else:
            print(f"Cleaned {len(df)} rows for key '{key}'")

        cleaned_data[key] = cleaned_df

        if to_json:
            filename = dir / f'{key}_clean.json'
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(cleaned_data[key].to_dict(orient='records'), f, ensure_ascii=False, indent=2)
            print(f"Saved {len(cleaned_data[key])} filtered posts from r/{key} to JSON")

    return cleaned_data