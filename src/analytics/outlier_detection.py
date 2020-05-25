from datetime import timedelta
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fbprophet import Prophet
from matplotlib import pyplot as plt


class TimeSeriesAnalyzer:
    """
    Class for outlier analysis of a single time series

    Parameters:
    :param old_data: Dataframe of dates (`ds`) and metric values (`y`) to train on
    :param new_data: Dataframe of dates (`ds`) and metric values (`y`) to test
    :param new_daterange: Dataframe of dates (`ds`) to predict for
    :param metadata: Dictionary with keys `dimension`, `item`, `metric`
    :param interval_multiplier: Coefficient defining relative accepted interval width.
        Higher = less sensitive

    Attributes:
    old_data: Dataframe of dates (`ds`) and metric values (`y`) to train on
    new_data: Dataframe of dates (`ds`) and metric values (`y`) to test
    new_daterange: Dataframe of dates (`ds`) to predict for
    metadata: Dictionary with keys `dimension`, `item`, `metric`
    interval_multiplier: Coefficient defining relative accepted interval width.
        Higher = less sensitive
    model: Fitted model for generating predictions
    deviations: Dataframe of deviations of test data
    """

    def __init__(
        self,
        old_data: pd.DataFrame,
        new_data: pd.DataFrame,
        new_daterange: pd.DataFrame,
        metadata: Dict[str, str],
        interval_multiplier: float = 1.5,
    ) -> None:
        self.old_data = old_data
        self.new_data = new_data
        self.new_daterange = new_daterange
        self.metadata = metadata
        self.interval_multiplier = interval_multiplier
        self.model = None
        self.deviations = None

    def find_deviations(self) -> None:
        """
        Fit model and generate deviations Dataframe
        """
        self.model = make_model(self.old_data)
        predictions = self.model.predict(self.new_daterange)
        self.deviations = get_deviations(
            predictions, self.new_data, self.interval_multiplier
        )

    def check_deviations(self) -> None:
        """
        Check if deviations Dataframe is computed
        """
        if not isinstance(self.deviations, pd.DataFrame):
            raise ValueError("Deviations not computed")

    def make_chart(self) -> plt.Figure:
        """
        Visualize deviations in a chart

        :return Matplotlib figure with a visualization
        """
        self.check_deviations()
        return create_chart(
            self.old_data, self.new_data, self.deviations, self.metadata
        )

    def get_filtered_deviations(self) -> pd.DataFrame:
        """
        Filter deviations -- only keep problematic dates

        :return Dataframe of problematic dates and deviations
        """
        self.check_deviations()
        return self.deviations.loc[lambda d: d["status"] != "ok"].assign(
            **self.metadata
        )[["ds", "metric", "dimension", "item", "status"]]

    def is_ok(self) -> bool:
        """
        Indicate if issues were found for any date

        :return: Boolean indicating if issues were found
        """
        self.check_deviations()
        n_lines_with_issues = len(self.deviations.loc[lambda d: d["status"] != "ok"])
        return n_lines_with_issues == 0

    def get_summary(self) -> pd.DataFrame:
        """
        Create a summary of results with metadata
        :return Dataframe of metadata and issue indication
        """
        return pd.DataFrame({**self.metadata, "ok": self.is_ok()}, index=[0])


def find_new_outliers(
    data: pd.DataFrame,
    reference_date: str,
    metrics: List[str],
    dimensions_items: Dict[str, str],
    dates_to_exclude: Optional[List[str]] = None,
) -> List[TimeSeriesAnalyzer]:
    """
    Find outliers for selected metrics and dimensions/items

    :param data: Input data to check outliers for
    :param reference_date: Cutoff date for training set.
    Model will use 365 days prior to this date to train.
    :param metrics: List of metrics to compute outliers for
    :param dimensions_items: Dimensions and items to compute outliers for,
    provided in a dictionary where keys are dimensions and values are lists of items
    :param dates_to_exclude: List of dates to exclude from training, optional.
    Use this if dataset still includes period of incorrect data.
    :return: List of fit TimeSeriesAnalyzers
    """
    dates_to_exclude = dates_to_exclude or []
    old_data, new_data = prepare(data, reference_date, dates_to_exclude)
    new_data_daterange = create_daterange_df(new_data)

    all_inputs = list()
    for metric in metrics:
        for dimension, items in dimensions_items.items():
            for item in items:
                if subset_exists(old_data, metric, dimension, item):
                    filters = {
                        "metric": metric,
                        "dimension": dimension,
                        "item": item,
                    }
                    old_data_item = slice_prophet(old_data, **filters)
                    new_data_item = slice_prophet(new_data, **filters)
                    all_inputs.append(
                        (old_data_item, new_data_item, new_data_daterange, filters)
                    )

    with Pool(5) as p:
        analyzers = p.map(_get_fit_analyzer, all_inputs)

    return analyzers


def _get_fit_analyzer(
    x: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, str]]
) -> TimeSeriesAnalyzer:
    """
    Helper function for initializing and fitting TimeSeriesAnalyzer objects

    :param x: Tuple of (old_data, new_data, new_daterange, metadata)
    """
    a = TimeSeriesAnalyzer(*x)
    a.find_deviations()
    return a


def make_model(training_data: pd.DataFrame) -> Prophet:
    """
    Initialize and train Prophet model using training data

    :param training_data: Dataframe of metric values with columns `ds` and `y`
    :return: Prophet model to predict with
    """
    m = Prophet(
        daily_seasonality=False,
        yearly_seasonality=False,
        interval_width=1,
        changepoint_range=1,
    )
    m.fit(training_data)
    return m


def get_deviations(
    predictions: pd.DataFrame, actuals: pd.DataFrame, interval_multiplier: float = 1.5
) -> pd.DataFrame:
    """
    Compute deviations of actuals from forecasts

    :param predictions: Dataframe of forecasted values generated by Prophet model
    :param actuals: Dataframe of true values, containing columns `ds` and `y`
    :param interval_multiplier: Coefficient to multiply width of allowed interval.
    Lower means more sensitive. Default = 1.5
    :return: Dataframe of all observations indicating their status
    (`ok`, `missing`, `above`, `below`).
    """

    def get_adjusted_bound(x: pd.Series, x_bound: pd.Series) -> pd.Series:
        return x + (x_bound - x) * interval_multiplier

    def get_status(d: pd.Series) -> np.ndarray:
        return np.where(
            d["y"].isna(),
            "missing",
            np.where(d["above"], "above", np.where(d["below"], "below", "ok")),
        )

    return (
        predictions.join(actuals.set_index("ds"), on="ds")
        .assign(
            upper_bound=lambda d: get_adjusted_bound(d["yhat"], d["yhat_upper"]),
            lower_bound=lambda d: get_adjusted_bound(d["yhat"], d["yhat_lower"]),
            above=lambda d: d["y"] > d["upper_bound"],
            below=lambda d: d["y"] < d["lower_bound"],
            status=get_status,
        )
        .sort_values("ds")[["ds", "y", "status", "yhat", "upper_bound", "lower_bound"]]
    )


def prepare(
    data: pd.DataFrame, reference_date: str, dates_to_exclude: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and prediction set

    Training set: period of one year preceding reference date, excluding specified dates.

    :param data: Dataframe of daily metrics
    :param reference_date: Date for split between training and predictions
    :param dates_to_exclude: Which dates to exclude from training set
    :return: Tuple of Dataframes for training set and prediction set
    """
    data = data.assign(ds=lambda d: pd.to_datetime(d["date"]))
    reference_date = pd.to_datetime(reference_date)
    old_data = data.loc[
        lambda d: (d["ds"] < reference_date)
        & ~d["ds"].isin(dates_to_exclude)
        & (d["ds"] >= reference_date - timedelta(days=365))
    ]
    last_reliable_date = data.loc[lambda d: d["ds"] >= reference_date]["ds"].max()
    new_data = data.loc[lambda d: d["ds"].between(reference_date, last_reliable_date)]

    return old_data, new_data


def subset_exists(data: pd.DataFrame, metric: str, dimension: str, item: str) -> bool:
    """
    Check if subset of selected parameters exists in given table
    :param data: Dataframe
    :param metric: Selected metric
    :param dimension: Selected dimension
    :param item: Selected item
    :return: Boolean indicating presence of parameters
    """
    return (
        (metric in data.columns)
        and (dimension in data["dimension"].unique())
        and (item in data["item"].unique())
    )


def create_daterange_df(
    new_data: pd.DataFrame, date_column: str = "ds"
) -> pd.DataFrame:
    """
    Create a Dataframe with a date range covering whole period
    :param new_data: Prediction Dataframe
    :param date_column: Column with date information
    :return: Dataframe with column `ds` spanning over whole period
    """
    return pd.DataFrame(
        {
            "ds": pd.date_range(
                new_data[date_column].min(), new_data[date_column].max(), freq="D"
            )
        }
    )


def slice_prophet(
    data: pd.DataFrame, dimension: str, item: str, metric: str,
) -> pd.DataFrame:
    """
    Slice data and transform to format used by Prophet

    :param data: Input dataframe
    :param dimension: Dimension
    :param item: Item
    :param metric: Metric
    :return: Dataframe for Prophet model
    """
    return data.rename(columns={metric: "y"}).loc[
        lambda d: (d["dimension"] == dimension) & (d["item"] == item)
    ]


def create_chart(
    old_data: pd.DataFrame,
    new_data: pd.DataFrame,
    deviations: pd.DataFrame,
    metadata: Dict[str, str],
) -> plt.Figure:
    """
    Create chart of deviations given training data, test data and predictions

    :param old_data: Dataframe of training data
    :param new_data: Dataframe of test data
    :param deviations: Dataframe of computed deviations
    :param metadata: Dictionary of `metric`, `dimension`, `item` metadata
    :return: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(old_data.sort_values("ds")["ds"], old_data.sort_values("ds")["y"])
    ax.scatter(old_data.sort_values("ds")["ds"], old_data.sort_values("ds")["y"])
    ax.fill_between(
        deviations["ds"],
        deviations["lower_bound"],
        deviations["upper_bound"],
        alpha=0.2,
    )
    ax.plot(deviations["ds"], deviations["yhat"], "--", alpha=0.5)
    ax.scatter(new_data["ds"], new_data["y"], c="red")
    ax.set_ylabel(metadata["metric"])
    ax.set_xlabel("Date")
    ax.set_xlim(left=old_data["ds"].max() - timedelta(days=60))
    ax.set_title(f"Predictions for {metadata['dimension']} / {metadata['item']}")
    return fig
