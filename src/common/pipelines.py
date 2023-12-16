from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from common import text_processing
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin


class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field

    def fit(self, X, y=None):   # noqa: N803
        return self

    def transform(self, X):  # noqa: N803
        return X[self.field]


def svm(min_df, max_df, ngram_range):
    pipeline = Pipeline(
        steps=[
            ('text_selection', TextSelector('text')),
            ('vectorize_text', TfidfVectorizer(preprocessor=text_processing.preprocess_text,
                                               tokenizer=text_processing.tokenize,
                                               token_pattern=None,
                                               stop_words=text_processing.stop_words(),
                                               sublinear_tf=True,
                                               ngram_range=ngram_range,
                                               min_df=min_df,
                                               max_df=max_df))
        ]
    )

    return pipeline
