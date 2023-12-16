import pandas as pd
from streamlit import session_state as ss
from sklearn.model_selection import train_test_split


def split(train_size, validation_size, test_size, seed=1234) -> pd.DataFrame:
    df = ss['dataframe']
    X = df.drop('target', axis=1)  # noqa: N806
    y = df['target'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size/100, random_state=seed)  # noqa: N806
    X_validation, X_test, y_validation, y_test = train_test_split(X_test,  # noqa: N806
                                                                  y_test,
                                                                  test_size=test_size/(test_size + validation_size),
                                                                  random_state=seed)

    return X_train, X_validation, X_test, y_train, y_validation, y_test
