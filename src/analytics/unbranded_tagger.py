import logging
import re
from itertools import chain
from typing import Iterable, List, Optional

import numpy as np
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
    to_label_tokens = np.array(cv_to_label.get_feature_names())

    logger.debug("transforming unbranded")
    cv_unbranded = CountVectorizer()
    unbranded_tokens_matrix = cv_unbranded.fit_transform(unbranded)
    unbranded_row_count = np.array(unbranded_tokens_matrix.sum(axis=0))[0, :]
    unbranded_tokens = np.array(cv_unbranded.get_feature_names())
    logger.debug(f"shape of unbranded row count: {unbranded_row_count.shape}")

    logger.debug("matching")
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

    logger.debug("filter columns")
    branded_tokens_mask = ~np.isin(to_label_tokens, unbranded_tokens)
    branded_row_counts = np.array(
        to_label_tokens_matrix[:, branded_tokens_mask].sum(axis=1)
    )[:, 0]

    return branded_row_counts == 0


def get_unique_tag_elements(tags: Iterable[str]) -> List[str]:
    """
    Get list of unique tag elements

    :param tags: Collection of tags including dashes, e.g. 'red-bull'
    :return: List of unique elements
    """
    tags_split = [b.split("-") for b in tags]
    tags_split_flat = chain.from_iterable(tags_split)
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
    word_matches = (
        texts_split[:, np.isin(cv.get_feature_names(), words)].sum(axis=1).transpose()
    )

    return np.array(word_matches)
