import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import utils

logger = logging.getLogger(__name__)


def compute_activity_stats(daily_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Compute activity statistics

    :param daily_stats: Dataframe of daily aggregated merchant data
    :return: Dataframe with merchant activity statistics
    """
    return (
        daily_stats.groupby(["vendor", "merchant"])
        .agg(
            active_days=("date", "count"),
            first_day=("date", "min"),
            last_day=("date", "max"),
            n_lines=("n_lines", "sum"),
            longest_gap=("date", utils.maximum_gap_in_days),
        )
        .reset_index()
        .assign(
            lifespan=lambda d: (d["last_day"] - d["first_day"]).dt.days + 1,
            activity=lambda d: d["active_days"] / d["lifespan"],
        )
    )


def compute_volatility(daily_stats: pd.DataFrame, min_days: int = 100) -> pd.DataFrame:
    """
    Compute metrics of volatility

    This measure describes volatility of merchant activity in terms of number of daily lineitems.
    For each merchant, we divide daily log(n_lines) by the respective average log(n_lines)
    for a given day of the week.

    :param daily_stats: Dataframe of daily aggregated merchant data (from BigQuery)
    :param min_days: Minimum amount of days merchants should be active
    :return: Dataframe with `relative_std` measure of volatility for each merchant
    """

    high_volume_merchants = (
        daily_stats.groupby(["vendor", "merchant"])
        .agg(active_days=("date", "count"))
        .reset_index()
        .loc[lambda d: d["active_days"] >= min_days]["merchant"]
    )

    daily_stats = daily_stats.assign(
        dow=lambda d: d["date"].dt.dayofweek.map(utils.WEEKDAYS),
        log_n_lines=lambda d: np.log(d["n_lines"]),
    )

    transaction_stats_weekday = (
        daily_stats.loc[lambda d: d["merchant"].isin(high_volume_merchants)]
        .groupby(["vendor", "merchant", "dow"])
        .agg(avg_log_transactions=("log_n_lines", "mean"))
        .assign(
            baseline=lambda d: d.groupby(["vendor", "merchant"])[
                "avg_log_transactions"
            ].transform("mean"),
            ratio_transactions=lambda d: d["avg_log_transactions"] / d["baseline"],
        )
    )

    daily_stats_normalized = daily_stats.join(
        transaction_stats_weekday["ratio_transactions"],
        how="left",
        on=["vendor", "merchant", "dow"],
    ).assign(
        normalized_log_transactions=lambda d: d["log_n_lines"] / d["ratio_transactions"]
    )

    volatility = (
        daily_stats_normalized.groupby(["vendor", "merchant"])
        .agg(
            std_transactions=("normalized_log_transactions", "std"),
            avg_transactions=("normalized_log_transactions", "mean"),
        )
        .assign(volatility=lambda d: d["std_transactions"] / d["avg_transactions"])
        .loc[lambda d: d["volatility"].notna()][["volatility"]]
    )

    return volatility


def get_adjusted_longest_gap(df: pd.DataFrame) -> pd.Series:
    """
    Compute longest period of inactivity, accounting for possible churn

    :param df: Dataframe of daily aggregated merchant data
    :return: Series of `longest_gap` values
    """
    dataset_last_day = df["last_day"].max()
    gap_at_the_end = (dataset_last_day - df["last_day"]).dt.days
    return df["longest_gap"].clip(lower=gap_at_the_end)


def compute_stable(
    df: pd.DataFrame,
    minimum_lifespan: int = 300,
    minimum_activity: float = 0.67,
    maximum_volatility: float = 0.67,
    maximum_gap: int = 21,
    **kwargs,
) -> np.ndarray:
    """
    Compute merchant stability from summary statistics

    :param df: Dataframe including merchant statistics
     `lifespan`, `activity`, `volatility` and `longest_gap`
    :param minimum_lifespan: Minimum merchant lifespan in days. Default 300.
    :param minimum_activity: Minimum activity (share of active days). Default 0.67.
    :param maximum_volatility: Maximum relative volatility. Default 0.67.
    :param maximum_gap: Longest allowed period of inactivity (in days). Default 21.
    :return: Numpy array of booleans indicating stability
    """
    allowed_settings = [
        "minimum_lifespan",
        "minimum_activity",
        "maximum_volatility",
        "maximum_gap",
    ]
    for kwarg in kwargs.keys():
        if kwarg not in allowed_settings:
            raise ValueError(
                f"Illegal argument `{kwarg}`. Allowed arguments: {allowed_settings}"
            )

    adjusted_longest_gap = get_adjusted_longest_gap(df)
    return (
        (df["lifespan"] >= minimum_lifespan)
        & (df["activity"] >= minimum_activity)
        & (df["volatility"] <= maximum_volatility)
        & (adjusted_longest_gap <= maximum_gap)
    )


def get_merchant_stats(
    daily_stats: pd.DataFrame, stability_settings: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Get dataframe of stability-related statistics per merchant

    :param daily_stats: Dataframe of daily aggregated merchant data (from BigQuery)
    :param stability_settings: Dictionary with arguments to pass to `compute_stable` (optional)
    :return: Dataframe with statistics
    """
    stability_settings = stability_settings or {}
    daily_stats = daily_stats.assign(date=lambda d: pd.to_datetime(d["date"]))
    activity_stats = compute_activity_stats(daily_stats)
    volatility = compute_volatility(daily_stats)

    cols_to_include = [
        "vendor",
        "merchant",
        "merchant_id",
        "lifespan",
        "active_days",
        "activity",
        "volatility",
        "longest_gap",
        "n_lines",
        "first_day",
        "last_day",
        "stable",
    ]

    return activity_stats.join(
        volatility, how="left", on=["vendor", "merchant"],
    ).assign(
        stable=lambda d: compute_stable(df=d, **stability_settings),
        merchant_id=utils.hash_merchant,
    )[
        cols_to_include
    ]


def make_stability_statistics(
    daily_stats: pd.DataFrame, years: Optional[List[int]] = None,
) -> pd.DataFrame:
    years = years or [2017, 2018, 2019]

    stats_by_year = list()
    for year in years:
        logger.info(f"Computing stable merchants for year {year}")
        stats_year = get_merchant_stats(
            daily_stats.loc[
                lambda d: d["date"].between(f"{year}-01-01", f"{year}-12-31")
            ]
        ).assign(year=year)[
            [
                "vendor",
                "merchant",
                "year",
                "stable",
                "lifespan",
                "activity",
                "volatility",
                "longest_gap",
            ]
        ]
        stats_by_year.append(stats_year)

    return pd.concat(stats_by_year)
