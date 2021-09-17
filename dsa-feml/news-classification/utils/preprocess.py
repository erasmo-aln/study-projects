import numpy as np

from nltk.corpus import stopwords

from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import constants


def read_dataset():

    dataset = load_files(container_path=constants.DATA_PATH, encoding=constants.ENCODING, decode_error='replace')

    X = dataset.data
    y = dataset.target

    return X, y


def set_stopwords():

    stop_words = set(stopwords.words(constants.STOP_WORDS_LANGUAGE))

    return stop_words


def split_dataset(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=constants.TEST_SIZE, random_state=constants.RANDOM_STATE)

    return X_train, X_test, y_train, y_test


def vectorize(X_train, X_test):

    stop_words = set_stopwords()
    vectorizer = TfidfVectorizer(norm=None, stop_words=stop_words, max_features=1000, decode_error='ignore')
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    return X_train_vectorized, X_test_vectorized
