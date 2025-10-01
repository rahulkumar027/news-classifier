# src/preprocessing.py
from sklearn.feature_extraction.text import TfidfVectorizer

def make_tfidf(min_df=1, ngram_range=(1, 2), max_df=0.9):
    """
    Create and return a configured TfidfVectorizer.
    """
    return TfidfVectorizer(
        strip_accents="unicode",
        sublinear_tf=True,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range
    )
