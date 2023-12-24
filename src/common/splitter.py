import numpy as np
import pandas as pd
from streamlit import session_state as ss
from sklearn.model_selection import train_test_split


def split(sample_count, train_size, validation_size, test_size, seed=1234) -> tuple[np.array,
                                                                                    np.array,
                                                                                    np.array,
                                                                                    np.array,
                                                                                    np.array,
                                                                                    np.array]:
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
