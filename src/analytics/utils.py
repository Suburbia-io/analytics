"""Various simple functions used throughout the package."""
import hashlib
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WEEKDAYS = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}


def maximum_gap_in_days(col: pd.Series) -> int:
    """Compute maximum gap in a series of dates
    (2020-01-01 - 2020-01-02 -> gap = 0 days)

    :param col: pd.Series of dates
    :return: greatest gap
    """
    return col.sort_values().diff().max().days - 1


def hash_merchant(df: pd.DataFrame) -> np.ndarray:
    """
    Create hashed merchant_id from `vendor` and `merchant` columns

    :param df: DataFrame containing `vendor` and `merchant` columns
    :return: Array of merchant_ids
    """
    concatenated = df["vendor"].astype(str) + "\n" + df["merchant"].astype(str)
    md5s = [hashlib.md5(val.encode()).hexdigest() for val in concatenated]
    return np.array(md5s)


def normalize(arr: np.ndarray) -> np.ndarray:
    """
    Scale input to the range [0,1].

    :param arr: an array of floats or integers
    :returns: all values in the original array scaled to [0,1]
    """
    if max(arr) - min(arr) == 0:
        logger.warning(
            "Normalize averted a div/0, the input data was:\n {0}".format(arr)
        )
        return np.ones(len(arr))
    return (arr - min(arr)) / (max(arr) - min(arr))


def ceil_int(a: np.ndarray, base: int = 1) -> np.ndarray:
    """
    Round int i up to the nearest multiple of "base".

    :param a: Numpy array of numbers
    :param base: the base to which the integer is rounded
    :returns: i rounded down to base
    """
    return (np.ceil(a / base) * base).astype(int)
