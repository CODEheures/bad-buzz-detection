import streamlit as st


def preprocess_text(df):
    st.write('Lower text...')
    df_copy = lowerize(df)
    st.write('Strip text...')
    df_copy = stripize(df_copy)
    return df_copy.values


def lowerize(df):  # noqa: N803
    df['text'] = df['text'].apply(lambda x: x.lower())
    return df


def stripize(df): # noqa N803
    df['text'] = df['text'].apply(lambda x: x.strip())
    return df
