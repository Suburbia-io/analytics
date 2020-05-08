import hashlib

import numpy as np
import pandas as pd


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
