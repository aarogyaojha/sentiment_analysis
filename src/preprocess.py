"""
preprocess.py
─────────────
Text preprocessing utilities shared between the notebook and any
standalone training scripts.

Usage
-----
    from src.preprocess import preprocess_tweet, build_tfidf

    df["processed"] = df["text"].apply(preprocess_tweet)
    vectorizer, X_train, X_test = build_tfidf(df_train["processed"], df_test["processed"])
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("stopwords", quiet=True)

_stemmer    = PorterStemmer()
_stop_words = set(stopwords.words("english"))


def preprocess_tweet(text: str) -> str:
    """
    Strip non-alpha characters, lowercase, remove stopwords, stem.

    This pipeline is used for TF-IDF input only.
    Do NOT apply it to DistilBERT input — BERT works better on raw text.

    Parameters
    ----------
    text : str
        Raw tweet or review text.

    Returns
    -------
    str
        Space-joined stemmed tokens with stopwords removed.
    """
    text   = re.sub(r"[^a-zA-Z]", " ", str(text))
    text   = text.lower()
    tokens = [_stemmer.stem(w) for w in text.split() if w not in _stop_words]
    return " ".join(tokens)


def build_tfidf(train_texts, test_texts, max_features: int = 100_000):
    """
    Fit a TF-IDF vectorizer on train_texts and transform both splits.

    Parameters
    ----------
    train_texts : iterable of str
    test_texts  : iterable of str
    max_features : int
        Vocabulary size cap. 100K covers the Sentiment140 vocabulary well.

    Returns
    -------
    vectorizer : TfidfVectorizer (fitted)
    X_train    : sparse matrix
    X_test     : sparse matrix
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train    = vectorizer.fit_transform(train_texts)
    X_test     = vectorizer.transform(test_texts)
    return vectorizer, X_train, X_test
