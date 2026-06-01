import numpy as np
import emoji

def build_freqs(tokens_list: list[list[str]], labels: np.ndarray) -> dict[tuple[str, int], int]:
    """
    Creates a frequency dictionary from a list of token lists and their labels.

    Args:
        tokens_list (list[list[str]]): A list containing lists of words (tokens).
        labels (np.ndarray): An array of the corresponding classifications (0 or 1).

    Returns:
        dict: A dictionary that maps (word, label) pairs to their absolute frequency.
    """
    freqs = {}
    for y, tokens in zip(labels, tokens_list):
        for word in tokens:
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1
    return freqs

def count_emojis(tokens: list[str]) -> int:
    """
    Counts the number of classic (ASCII) and modern (Unicode) emojis in a list of tokens.

    Args:
        tokens (list[str]): The list of words to be analysed.

    Returns:
        int: The total number of emojis found.
    """
    classic_faces = [
        ":)", ":-)", ":D", ":-D", ":(", ":-(", ":|", ":-|", 
        ":P", ":-P", ";)", ";-)", ":O", ":-O", ":/", ":-/", 
        ":\\", ":-\\", ":*", ":-*", ":'(", ":'-(", "XD", "xD", ">:("
    ]
    modern_faces = ["^_^", "-_-", ">_<", "T_T", "UwU", "OwO", "o_O", "O_o", "-__-"]
    ascii_faces = classic_faces + modern_faces
    
    unicode_emojis = sum(1 for token in tokens if emoji.is_emoji(token))
    ascii_emojis = len({face for face in ascii_faces if face in tokens})
    
    return unicode_emojis + ascii_emojis

def extract_features(tokens: list[str], freqs: dict[tuple[str, int], int]) -> np.ndarray:
    """
    Extracts numerical features from a list of tokens based on frequencies.

    Args:
        tokens (list[str]): The tokenised words of a document.
        freqs (dict): The trained frequency dictionary.

    Returns:
        np.ndarray: An 8-dimensional array containing the extracted features (including bias).
    """
    pos = 0
    neg = 0
    for word in tokens:
        if (word, 1) in freqs: pos += 1
        if (word, 0) in freqs: neg += 1
            
    pos_score = sum([freqs.get((word, 1), 0) for word in tokens])
    neg_score = sum([freqs.get((word, 0), 0) for word in tokens])
    
    emoji_count = count_emojis(tokens)
    tweet_length = len(tokens)
    avg_word_len = np.mean([len(w) for w in tokens]) if tokens else 0

    return np.array([1, pos, neg, pos_score, neg_score, emoji_count, tweet_length, avg_word_len])