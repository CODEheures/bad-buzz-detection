import numpy as np
import pandas as pd
from streamlit import session_state as ss
from sklearn.model_selection import train_test_split


def split(sample_count: int,
          train_size: int,
          validation_size: int,
          test_size: int,
          seed=1234) -> tuple[np.array,
                              np.array,
                              np.array,
                              np.array,
                              np.array,
                              np.array]:
    """Split the ss['dataframe'] dataframe into train/validation/test

    Args:
        sample_count (int): The number of sample to keep. Ex: 600
        train_size (int): The %part of train. Ex: 50 to use 50% for train
        validation_size (int): The %part of validation. Ex: 25 to use 25% for validation
        test_size (int): The %part of test. Ex: 25 to use 25% for test
        seed (int, optional): the seed to replication same split. Defaults to 1234.

    Returns:
        tuple[np.array, np.array, np.array, np.array, np.array, np.array]:
        In order X_train, X_validation, X_test, y_train, y_validation, y_test
    """
    df: pd.DataFrame = ss['dataframe']
    df_sample = df.sample(n=sample_count, random_state=seed)
    X = df_sample.text.to_numpy()  # noqa: N806
    y = df_sample.target.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X,  # noqa: N806
                                                        y,
                                                        train_size=train_size/100,
                                                        stratify=y,
                                                        random_state=seed)
    X_validation, X_test, y_validation, y_test = train_test_split(X_test,  # noqa: N806
                                                                  y_test,
                                                                  stratify=y_test,
                                                                  test_size=test_size/(test_size + validation_size),
                                                                  random_state=seed)

    return X_train, X_validation, X_test, y_train, y_validation, y_test
