from src.common import text_processing


def test_stop_words():
    stop_words = text_processing.stop_words()
    assert type(stop_words) is list
    assert len(stop_words) > 0


def test_preprocess():
    text = ' HELLO this Is a test '
    result = 'hello this is a test'
    preprocessed = text_processing.preprocess_text(text=text)
    assert preprocessed == result


def test_tokenize_lemmatize():
    text = 'hello this is a test'
    tokens = text_processing.tokenize_lemmatize(text=text)
    assert type(tokens) is list
    assert len(tokens) == 4
    assert 'hello' in tokens
    assert 'this' in tokens
    assert 'be' in tokens
    assert 'test' in tokens


def test_tokenize_stemming():
    text = 'hello this is a test'
    tokens = text_processing.tokenize_stemming(text=text)
    assert type(tokens) is list
    assert len(tokens) == 4
    assert 'hello' in tokens
    assert 'thi' in tokens
    assert 'is' in tokens
    assert 'test' in tokens
