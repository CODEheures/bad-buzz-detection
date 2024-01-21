from nltk.corpus import stopwords
from nltk import RegexpTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
from transformers import AutoTokenizer
from torch.utils.data import Dataset

bert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokenize_bert_max_length = 50


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


def tokenize_lemmatize(text: str) -> list[str]:
    """Tokenizer with lemmatizer

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
    tokens = [token for token in tokens if len(token) > 1]

    return tokens


def tokenize_stemming(text: str) -> list[str]:
    """Tokenizer with stemming

    Args:
        text (str): The text to tokenize

    Returns:
        list[str]: Tokens
    """
    # Tokenization
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)

    # Lematization
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    # Keep tokens with length > 2
    tokens = [token for token in tokens if len(token) > 1]

    return tokens


def tokenize_bert(dataset: Dataset):
    # Pad/truncate each text to 512 tokens. Enforcing the same shape
    # could make the training faster.
    return bert_tokenizer(
        dataset["tweets"],
        padding="max_length",
        truncation=True,
        max_length=tokenize_bert_max_length,
    )
