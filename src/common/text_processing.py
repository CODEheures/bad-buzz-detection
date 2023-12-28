from nltk.corpus import stopwords
from nltk import RegexpTokenizer
from nltk.stem import WordNetLemmatizer


def stop_words() -> list[str]:
    """Get list of english stop word

    Returns:
        list[str]: The english stop words
    """
    return stopwords.words('english')


def preprocess_text(text: str) -> str:
    """Preprocess a text by lowering and stripping

    Args:
        text (str): the text to process

    Returns:
        str: The processed text
    """
    return text.lower().strip()


def tokenize(text: str) -> list[str]:
    """Tokenizer

    Args:
        text (str): The text to tokenize

    Returns:
        list[str]: Tokens
    """
    # Tokenization
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)

    # Lematization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word, pos='a') for word in tokens]
    tokens = [lemmatizer.lemmatize(word, pos='v') for word in tokens]
    tokens = [lemmatizer.lemmatize(word, pos='n') for word in tokens]

    # Keep tokens with length > 2
    tokens = [token for token in tokens if len(token) > 2]

    return tokens
