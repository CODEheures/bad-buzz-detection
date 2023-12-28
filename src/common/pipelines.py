from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from common import text_processing


def svm(min_df: int,
        max_df: float,
        ngram_range: tuple,
        C: float,   # noqa= N803
        degree: int,
        kernel,
        gamma: float,
        seed=1234) -> Pipeline:
    """Create a personalized SVM pipeline

    Args:
        min_df (int): The word must be appear at least 'min_df' times to be keeped
        max_df (float): The word must be in the 'max_df'% of the dict to be keeped
        ngram_range (tuple): To create ngrams. Ex: (1,2) to create mono and bi-grams
        C (float): C param for SVM
        kernel ({'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}): The kernel to use on SVM
        gamma (float): Gamma param for SVM
        seed (int, optional): Seed to replay SVM. Defaults to 1234.

    Returns:
        Pipeline: The SVM pipeline
    """
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
