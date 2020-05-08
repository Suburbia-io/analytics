import logging
import re
from typing import Iterable, List, Optional

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

logger = logging.getLogger(__name__)


def get_unbranded(
    to_label: np.ndarray,
    unbranded: np.ndarray,
    tokens_to_include: Optional[List[str]] = None,
    tokens_to_exclude: Optional[List[str]] = None,
    skip_numbers: bool = True,
    min_unbranded_rows: int = 5,
) -> np.ndarray:
    """
    Label items as unbranded given a set of existing unbranded rows

    This algorithm finds new unbranded rows by looking at texts already labeled
    as unbranded. Using the assumption that a description of an unbranded item
    will consist only of words unrelated to brands, we build a list
    of "unbranded" words. Then, an item is labeled as "unbranded" if
    its description only consists of these "unbranded" words.

    :param to_label: Numpy array of raw_texts to label
    :param unbranded: Numpy array of raw_texts already labeled as unbranded
    :param tokens_to_include: Extra tokens should be considered as unbranded
    :param tokens_to_exclude: Extra tokens that *may* appear in unbranded tags
    but should *not* be considered as unbranded
    :param skip_numbers: Consider digit-only tokens as unbranded
    :param min_unbranded_rows: Minimal number of unbranded fingerprints
    to consider for an "unbranded" token. Lower number will increase total
    number of new identified unbranded rows but likely also increase false
    positive rate (given incorrect labels).
    :return: Numpy array of boolean values of same shape as `to_label`.
    """
    logger.debug("transforming to_label")
    cv_to_label = CountVectorizer()
    to_label_tokens_matrix = cv_to_label.fit_transform(to_label)
    to_label_token_count = np.array(to_label_tokens_matrix.sum(axis=1))[:, 0]

    logger.debug("transforming unbranded")
    cv_unbranded = CountVectorizer()
    unbranded_tokens_matrix = cv_unbranded.fit_transform(unbranded)
    unbranded_row_count = np.array(unbranded_tokens_matrix.sum(axis=0))[0, :]
    logger.debug(unbranded_row_count.shape)

    logger.debug("matching")
    to_label_tokens = np.array(cv_to_label.get_feature_names())
    unbranded_tokens = np.array(cv_unbranded.get_feature_names())
    # only include tokens present in at least `min_unbranded_rows` fingerprints
    unbranded_tokens = unbranded_tokens[unbranded_row_count >= min_unbranded_rows]

    if tokens_to_include:
        unbranded_tokens = np.hstack([unbranded_tokens, np.array(tokens_to_include)])

    if skip_numbers:
        number_tokens = np.array([w for w in to_label_tokens if re.match(r"\d+", w)])
        unbranded_tokens = np.hstack([unbranded_tokens, number_tokens])

    if tokens_to_exclude:
        unbranded_tokens = unbranded_tokens[
            ~np.isin(unbranded_tokens, tokens_to_exclude)
        ]

    # create a matrix indicating matches between each pair
    # of `to_label_tokens` and `unbranded_tokens`
    logger.debug("creating filter matrix")
    filter_matrix = create_indicator_matrix(to_label_tokens, unbranded_tokens)

    logger.debug(f"filter matrix size: {filter_matrix.shape}")
    to_label_tokens_unbranded_matrix = to_label_tokens_matrix * filter_matrix
    # number of unbranded tokens in each text from `to_label`
    to_label_unbranded_token_count = np.array(
        to_label_tokens_unbranded_matrix.sum(axis=1)
    )[:, 0]

    return to_label_token_count == to_label_unbranded_token_count


def create_indicator_matrix(x: np.ndarray, y: np.ndarray,) -> sparse.csc_matrix:
    """
    Create a diagonal matrix indicating if elements of `x` are present in `y`

    :param x: Array
    :param y: Array of same type as x
    :return: Sparse diagonal matrix of occurrences of `x` in `y`
    """
    matches = np.isin(x, y).astype(int)
    return sparse.diags(matches).tocsc()


def get_unique_tag_elements(tags: Iterable[str]) -> List[str]:
    """
    Get list of unique tag elements

    :param tags: Collection of tags including dashes, e.g. 'red-bull'
    :return: List of unique elements
    """
    tags_split = [b.split("-") for b in tags]
    tags_split_flat = [item for sublist in tags_split for item in sublist]
    return list(set(tags_split_flat))


def get_n_word_matches(texts: np.ndarray, words: np.ndarray) -> np.ndarray:
    """
    Count number of word matches in texts

    :param texts: Array of texts
    :param words: List of words to match against
    :return: Array indicating number of matching words, of same shape as `texts`
    """
    cv = CountVectorizer()
    texts_split = cv.fit_transform(texts)

    filter_matrix = create_indicator_matrix(cv.get_feature_names(), np.array(words))
    texts_with_words = texts_split * filter_matrix

    word_matches = np.array(texts_with_words.sum(axis=1))[:, 0]
    return word_matches
