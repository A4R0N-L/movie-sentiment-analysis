import re
import string
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

def process_tweet(tweet: str) -> list[str]:
    """
    Bereinigt einen Rohtext durch Entfernen von Rauschen, Stoppwörtern und Satzzeichen.
    Wendet Stemming und logische Negation (z.B. 'NOT_good') an.

    Args:
        tweet (str): Der rohe Eingabetext.

    Returns:
        list[str]: Eine Liste von bereinigten und normalisierten Tokens.
    """
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    
    # Rauschen entfernen (Ticker, RTs, Hyperlinks)
    tweet = re.sub(r"\$\w*", "", tweet)
    tweet = re.sub(r"^RT[\s]+", "", tweet)
    tweet = re.sub(r"https?:\/\/.*[\r\n]*", "", tweet)
    
    tokenizer = TweetTokenizer(preserve_case=True, strip_handles=True, reduce_len=False)
    tokens = tokenizer.tokenize(tweet)
    
    negation_words = {
        "not", "no", "never", "n't", "ain't", "wasn't", "isn't", 
        "don't", "doesn't", "didn't", "can't", "couldn't", "won't", 
        "wouldn't", "shouldn't"
    }
    processed_tokens = []
    negate_next = False

    for token in tokens:
        token_lower = token.lower()
        if token in string.punctuation:
            continue
        if token_lower in negation_words:
            negate_next = True
            continue
            
        if token_lower not in stop_words and token not in string.punctuation:
            is_valid_word = re.match(r'^[a-zA-Z0-9_]+$', token)
            if is_valid_word:
                token = stemmer.stem(token)
            if negate_next and is_valid_word:
                token = f"NOT_{token}"
                negate_next = False
            if token:
                processed_tokens.append(token)
                
    return processed_tokens

def process_movie_review(tokens: list[str]) -> list[str]:
    """
    Angepasstes Preprocessing für bereits tokenisierte Movie Reviews.
    Entfernt Stoppwörter sowie Satzzeichen und wendet den PorterStemmer an.

    Args:
        tokens (list[str]): Die bereits vom NLTK-Korpus aufgeteilten Wörter.

    Returns:
        list[str]: Eine Liste von bereinigten und normalisierten Tokens.
    """
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    processed = []
    
    for word in tokens:
        word = word.lower()
        if word not in stop_words and word not in string.punctuation:
            processed.append(stemmer.stem(word))
            
    return processed