import numpy as np
import emoji

def build_freqs(tokens_list: list[list[str]], labels: np.ndarray) -> dict[tuple[str, int], int]:
    """
    Erstellt ein Frequenz-Wörterbuch aus einer Liste von Token-Listen und deren Labels.

    Args:
        tokens_list (list[list[str]]): Eine Liste, die wiederum Listen von Wörtern (Tokens) enthält.
        labels (np.ndarray): Array der zugehörigen Klassifikationen (0 oder 1).

    Returns:
        dict: Ein Wörterbuch, das Paare aus (Wort, Label) auf ihre absolute Häufigkeit mappt.
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
    Zählt die Anzahl klassischer (ASCII) und moderner (Unicode) Emojis in einer Token-Liste.

    Args:
        tokens (list[str]): Die zu analysierende Wortliste.

    Returns:
        int: Die Gesamtsumme der gefundenen Emojis.
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
    Extrahiert numerische Merkmale (Features) aus einer Token-Liste basierend auf Frequenzen.

    Args:
        tokens (list[str]): Die bereinigten Wörter eines Dokuments.
        freqs (dict): Das trainierte Frequenz-Wörterbuch.

    Returns:
        np.ndarray: Ein 8-dimensionaler Vektor mit den extrahierten Merkmalen (inklusive Bias).
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