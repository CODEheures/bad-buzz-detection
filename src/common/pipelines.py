from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from common import text_processing
from sklearn.base import BaseEstimator, TransformerMixin


class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field

    def fit(self, X, y=None):   # noqa: N803
        return self

    def transform(self, X):  # noqa: N803
        return X[self.field]


def svm(min_df,
        max_df,
        ngram_range,
        C,   # noqa= N803
        degree,
        kernel,
        gamma,
        seed=1234):
    pipeline = Pipeline(
        steps=[
            ('vectorize_text', TfidfVectorizer(preprocessor=text_processing.preprocess_text,
                                               tokenizer=text_processing.tokenize,
                                               token_pattern=None,
                                               stop_words=text_processing.stop_words(),
                                               sublinear_tf=True,
                                               ngram_range=ngram_range,
                                               min_df=min_df,
                                               max_df=max_df)),
            ('modelize', SVC(C=C,
                             degree=degree,
                             kernel=kernel,
                             gamma=gamma,
                             random_state=seed,
                             ))
        ]
    )

    return pipeline
