"""
Generate and process data in Suburbia's simulated environment.

This code is written for educational and documentation purposes.
This implementation has not been used at scale.
We used a pipeline developed in go for this.

"""
import re
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats

from .utils import ceil_int, normalize

DictList = List[Dict[str, Union[str, List[Tuple[str, float]]]]]


def get_example_merchants() -> DictList:
    """
    Return example merchant information as obtained from POS systems.

    Example data from Point-of-Sales sytems used in bars and restaurants about
    the merchants (the actual bars and restaurants) and their menu items.

    The returned list has the following structure:
    list [merchant_item1, merchant_item2,...] one item for each merchant
    The information per merchant is a dictionary with key, values of:
    merchant: the merchant's identifier
    data_source: the source that delivers the data
    location: the location or country in Alpha-2 code
    items: a list of tuples with (menu_item, price)


    :returns: dictionary with example POS data
    """
    return [
        {
            "merchant": "merchant_1",
            "data_source": "A",
            "location": "DE",
            "items": [
                ("flasche heineken bier 0.5", 4),
                ("coca-cola", 3),
                ("pizza", 12),
                ("tee mit milch", 1),
                ("milch", 3),
            ],
        },
        {
            "merchant": "merchant_2",
            "data_source": "A",
            "location": "DE",
            "items": [
                ("bier klein", 3),
                ("kaffee mit milch", 1),
                ("kaffee mit sucker", 9),
                ("tee mit sucker", 1),
            ],
        },
        {
            "merchant": "merchant_3",
            "data_source": "B",
            "location": "NL",
            "items": [
                ("alcoholvrijbier", 5),
                ("bepsi-cola", 2),
                ("fles wijn", 25),
                ("rode wijn", 8),
            ],
        },
        {
            "merchant": "merchant_4",
            "data_source": "B",
            "location": "NL",
            "items": [("cola can", 5), ("pizza margeritha", 9), ("fles witte", 5)],
        },
        {
            "merchant": "merchant_5",
            "data_source": "B",
            "location": "BE",
            "items": [("friet", 6), ("bier stella", 7), ("witte wijn", 1)],
        },
    ]


def get_example_category_map() -> Dict[str, str]:
    """
    Return example category mapping based on simple regex.

    The mapping is from patterns found in menu_items to a category.
    The returned dictionary {key, value} has the following structure:
    key is the regex search pattern
    value is the value this pattern is mapped to

    :returns: dictionary with example category mapping
    """
    return {
        "bier": "drinks/beer",
        "cola": "drinks/cola",
        "friet": "food/fries",
        "hamburger": "food/hamburger",
        "kaffee": "drinks/coffee",
        "pizza": "food/pizza",
        "tee": "drinks/tea",
    }


def get_example_brand_map() -> Dict[str, str]:
    """
    Return example brand mapping based on simple regex.

    The mapping is from patterns found in menu_items to the items brand.
    The returned dictionary {key, value} has the following structure:
    key is the regex search pattern
    value is the value this pattern is mapped to

    :returns: dictionary with example brand mapping
    """
    return {
        "heineken": "heineken",
        "coca.cola": "coca-cola",
        "red.bull": "red bull",
        "pizza": "unbranded",
        "tee": "unbranded",
        "hamburger": "unbranded",
        "stella": "stella artois",
        "tee mit milch": "unbranded",
        "kaffee mit sucker": "unbranded",
        "rode wijn": "unbranded",
        "fles witte": "unbranded",
    }


def expand_column(df: pd.DataFrame, col: str, names: List[str] = None) -> pd.DataFrame:
    """
    Expand a DataFrame column containing lists to multiple columns.

    Expands a DataFrame column where each row contains a same-sized lists
    Here it serves to expand the [item, price] list into two columns.

    :param df: dataframe with at least one column named the same as param:col
    :param col: string with the name of the column that should be expanded
    :param names: list of string with the names of the new columns.
        len(names) should be the same lenth as the lists in df.col

    :returns: pandas DataFrame containing the same information as the input,
        structured as one item per row.

    """
    expanded = df[col].apply(pd.Series, index=names)
    return df.join(expanded).drop(columns=col)


def create_merchants_df(merchants: DictList) -> pd.DataFrame:
    """
    Create a dataframe based on a merchant information dictionary.

    Creates a row per menu item  with price and all other merchant information.

    :param merchants: a list of dicts of merchant information the structure is
        described in the get_example_merchants() method.
    :returns: pandas DataFrame containing the same information as the input,
        structured as one item per row.
    """
    return (
        pd.DataFrame(merchants)
        .explode("items")
        .reset_index(drop=True)
        .pipe(expand_column, "items", ["item", "unit_price"])
    )


def get_trend(days: int) -> np.ndarray:
    """
    Return an array with the sales trend as seen in restaurant sales.

    The trend is calculated from a yearly growth. This growth is drawn
    at random from a normal distribution such that on average ther will be a
    5% growth per year. The statistics are based on "it looks good" no deeper
    statistical rigor. Returned will be an array with growth per day starting
    at day 1 with value 1 and on average ending with 1.05 on day 365.

    :param days: an integer with the number of days that should be returned.
    :returns: numpy ndarray with daily growth.
    """
    mu, sigma = 0.05, 0.1
    growth = np.random.normal(mu, sigma)
    growth = 1 + growth * days / 365
    trend = np.linspace(1, growth, days)

    return trend


def get_seasonality(days: int) -> np.ndarray:
    """
    Return an array with the weekly seasonality as seen in restaurant sales.

    The seasonality is generated by sampling the standard normal probability
    density function.  the pattern is repeated every 7 days. The exact shape
    of the pdf is determined by a random shape. The statistics are based on
    "it looks good" no deeper statistical rigor. However, it resembles consumer
    behaviour with weekends having a sales peak while weekdays are more slow.

    :param days: an integer with the number of days that should be returned.
    :returns: numpy ndarray with weekly seasonality growth of length days.
    """
    mu, sigma = 1.5, 0.75
    shape = np.random.normal(mu, sigma)
    shape = mu if shape <= 0 else shape
    weekly = stats.norm.pdf(range(7), 4.6, shape)
    seasonality = np.tile(weekly, 1 + days // 7)[:days]

    return normalize(seasonality)


def get_noise(days: int) -> np.ndarray:
    """
    Return an array with the sales noise as seen in restaurant sales.

    The noise is generated by drawing random normally distributed samples.
    The statistics are based on "it looks good" no deeper statistical rigor.
    However, it resembles consumer behaviour with some days being more or less
    active based on weather, events holidays etc.

    :param days: an integer with the number of days that should be returned.
    :returns: numpy ndarray with sales "noise" of length days.
    """
    mu, sigma = 0, 0.1
    return normalize(np.random.normal(mu, sigma, days))


def generate_total_sales(days: int) -> np.ndarray:
    """
    Return an array with the simulated total sales as seen in restaurant sales.

    The daily total sales is generated by combining:
        trend, (weekly) seasonality and noise

    :param days: an integer with the number of days that should be returned.
    :returns: numpy ndarray with simulated daily total sales.
    """
    scale_ts = np.random.normal(1, 0.1)
    scale_noise = np.random.normal(0.5, 0.05)
    scale_all = np.random.normal(500, 50)

    trend = get_trend(days)
    seasonality = get_seasonality(days)
    noise = get_noise(days)

    sales = scale_all * (scale_ts * trend * seasonality + scale_noise * noise)

    return sales.astype(int)


def create_total_sales_df(index: int, days: int) -> pd.DataFrame:
    """
    Return a DataFrame with the simulated sales as for one merchat's menu item.

    The daily total sales are generated in generate_total_sales().
    This method returns a DataFrame with 3 columns:
        index: the index used to indicate the item the sales are generated for
        date: [0,...,days-1] the date of the simulated sales
        total_sales: the simulated daily total sales (for item index)

    :param index: integer indicating a unique menue item sold by a merchant
    :param days: integer with the number of days that should be returned.
    :returns: pandas DataFrame with simulated daily total sales, and the dates.
    """
    return pd.DataFrame(
        {
            "index": index,
            "date": np.arange(days),
            "total_sales": generate_total_sales(days),
        }
    ).set_index("index")


def add_all_total_sales(df: pd.DataFrame, days: int) -> pd.DataFrame:
    """
    Add simulated total sales for all merchants and all menue items to df.

    Join the DataFrames per item as created by create_total_sales_df() to
    the original DataFrame df. This means that df goes from one row per menu
    item to "days" number of rows per menu item.

    :param df: pandas DataFrame as created using create_merchants_df()
    :param days: integer with the number of days that should be simulated.
    :returns: pandas DataFrame with simulated daily total sales, together with
        merchant level information and item and price.
    """
    all_item_sales = pd.concat(
        [create_total_sales_df(x.Index, days=days) for x in df.itertuples()]
    )
    return df.join(all_item_sales).reset_index(drop=True)


def generate_quantities(total_sales: int) -> np.ndarray:
    """
    Generate individual item sales based on an item's total sales for a day.

    Each individual sale consists of the quantity of items being sold. Lower
    quantities are more common (actually qunatities 1 & 2 are equally common)
    with number of occurances halving for every extra unit per quatity. To
    simulate this we chose an exponential distribution to sample from; because
        1: we only get positive numbers and
        2: the lower the number the more occurances.

    :param total_sales: integer with the total sales for the item for a day
    :returns: numpy ndarray with quantities of items sold, the quantities sum
        to roughly total_sales.
    """
    quantities = np.ceil(np.random.exponential(scale=1.0, size=total_sales)).astype(int)
    quantities = quantities[np.cumsum(quantities) <= total_sales]
    return quantities


def create_quantities_df(index: int, total_sales: int) -> pd.DataFrame:
    """
    Return a DataFrame with the simulated item quantities sold.

    The individual item sales based on an item's total sales for a day are
    generated in generate_quantities().
    This method returns a DataFrame with one column and an index that indicates
    the item the sales are generated for. The quantity column consists of
    the quantity involved in one "receipt line". e.g. the following consists of
    3 receipt lines with quantities 2,2,1
        "2 Hamburgers"
        "2 Beer"
        "1 Cola"
    :param index: integer indicating a unique menue item, merchant, day
    :param total_sales: integer with the total sales for the item for a day
    :returns: pandas DataFrame with one row per receip line with sold quantity
    """
    return pd.DataFrame(
        {"index": index, "quantity": generate_quantities(total_sales)}
    ).set_index("index")


def add_all_quantities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simulated "receipt line" sales for all merchants, items, days.

    Join the DataFrames per merchhant, item, day as created by
    create_quntities_df() to the original DataFrame df. This means that df goes
        from: day * len(merchant menu item)
        to:   day * len(merchant menu item) * len(receipt lines per item)

    :param df: pandas DataFrame as created using add_all_total_sales()
    :returns: pandas DataFrame with simulated receipt line quantity for sales,
        together with total sales, merchant level information, item and price.
    """
    item_quantities = pd.concat(
        [
            create_quantities_df(index=x.Index, total_sales=x.total_sales)
            for x in df.itertuples()
        ]
    )
    return df.join(item_quantities).reset_index(drop=True)


def generate_date_merch_ids(df: pd.DataFrame) -> pd.Series:
    """
    Generate a unique receipt identifier for a merchant, date combination.

    In bars and restaurants you don't order one thing at a time but a
    couple of items together end up on a receipt. The receipt_id groups
    multiple receipt item lines together. e.g.
        "2 Hamburgers"
        "2 Beer"
        "1 Cola"
    The receipt_ids are generated per merchant per day. We sample random
    numbers to generate receipt_ids such that on average there will be 2 items
    per receipt(+1 such that we always have at least 1 number to choose from).
    This prefers generating short receipts over longer ones.

    :param df: pandas DataFrame as generated by the group by in
        generate_receipt_ids()

    :returns: pandas Series with unique ids.
    """
    avg_items_per_receipt = 2
    ids = np.random.choice(
        np.arange(1 + len(df) // avg_items_per_receipt), size=len(df)
    )
    date_merch = "_".join([str(dm) for dm in df.name])
    return pd.Series(ids).astype(str) + date_merch


def generate_receipt_ids(df: pd.DataFrame) -> np.ndarray:
    """
    Combine generated receipt ids per merchant, date.

    :param df: pandas DataFrame as created using add_all_quantities(),
        minimally with columns "merchant" and "date"
    :returns: numpy ndarray with unique ids stacked for all merchants and days.
    """
    receipt_ids = df.groupby(["date", "merchant"]).apply(generate_date_merch_ids)
    return np.hstack(receipt_ids)


def create_cpg_input_df(merchants: DictList, days: int = 30) -> pd.DataFrame:
    """
    Create a DataFrame similar to input data seen in Suburbia's cpg-data.

    The input data is generated using all methods described above.

    :param merchants: a list of dicts as created in get_example_merchants()
    :param days: integer with the number of days that should be simulated.
    :returns: a DataFrame with simulated suburbia input data, with columns:
        "date", "reporting_date", "batch", "data_source", "merchant",
        "location", "receipt_id", "line_id", "item", "unit_price", "quantity",
        "volume_eur"
    """
    return (
        create_merchants_df(merchants)
        .pipe(add_all_total_sales, days=days)  # add daily item sales
        .pipe(add_all_quantities)  # add individual quantities
        .reset_index(drop=True)
        .assign(
            volume_eur=lambda d: d["unit_price"] * d["quantity"],
            reporting_date=lambda d: d["date"]
            + np.round(np.random.exponential(1)).astype(int),
            batch=lambda d: d["reporting_date"]
            .map(lambda x: ceil_int(x, 7))
            .astype(int),
            receipt_id=generate_receipt_ids,
            line_id=lambda d: d.index,
        )
        .sort_values(by=["date", "data_source", "merchant", "line_id"])
        .reset_index(drop=True)[
            [
                "date",
                "reporting_date",
                "batch",
                "data_source",
                "merchant",
                "location",
                "receipt_id",
                "line_id",
                "item",
                "unit_price",
                "quantity",
                "volume_eur",
            ]
        ]
    )


def expand_map(compact_map: Dict[str, str], all_items: np.ndarray) -> Dict[str, str]:
    """
    Expand a compact representation of a mapping to an expanded version.

    :param compact_map: a dictonary as created in get_example_*_map()
    :param all_items: a numpy ndarray with all unique items to be mapped
    :returns: a map for all_items to their mapped value
    """
    return {
        item: value
        for pattern, value in compact_map.items()
        for item in all_items
        if re.search(pattern, item)
    }


def clean_data(
    df: pd.DataFrame,
    expanded_cat_map: Dict[str, str],
    expanded_brand_map: Dict[str, str],
) -> pd.DataFrame:
    """
    Clean input data by mapping items to their mapped values.

    :param df: a pandas DataFrame as created by create_input_df()
    :param expanded_cat_map: an expanded mapping for categories as obtained
        from expand_map()
    :param expanded_brand_map: an expanded mapping for brands as obtained from
        expand_map()
    :returns: a DataFrame with 2 extra columns "category" and "brand" where
        all_items are mapped to their mapped "category" and "brand" value
    """
    return df.assign(
        category=lambda d: d["item"].map(expanded_cat_map).fillna("unknown"),
        brand=lambda d: d["item"].map(expanded_brand_map).fillna("unknown"),
    )


def create_cpg_df(
    merchants: DictList,
    compact_cat_map: Dict[str, str],
    compact_brand_map: Dict[str, str],
    days: int = 30,
) -> pd.DataFrame:
    """
    Create a DataFrame similar to final data seen in Suburbia's cpg-data.

    This will produce the simulated enriched dataset, including simulating
    input data and cleaning that data.

    :param merchants: a list of dicts as created in get_example_merchants()
    :param compact_cat_map: a dictonary as created in get_example_cat_map()
        with a mapping from items to categories.
    :param compact_brand_map: a dictonary as created in get_example_brand_map()
        with a mapping from items to brands.
    :param days: integer with the number of days that should be simulated.
    :returns: a DataFrame with simulated suburbia enriched data, with columns:
        "date", "reporting_date", "batch", "data_source", "merchant",
        "location", "receipt_id", "line_id", "item", "unit_price", "quantity",
        "volume_eur", "category", "brand"
    """
    df = create_cpg_input_df(merchants, days)
    expanded_cat_map = expand_map(
        compact_map=compact_cat_map, all_items=df["item"].unique()
    )
    expanded_brand_map = expand_map(
        compact_map=compact_brand_map, all_items=df["item"].unique()
    )
    return clean_data(df, expanded_cat_map, expanded_brand_map)


def problem1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a simulated problem (1) with the input data.

    Problem: a broken mapping because of change in source data; 1 batch is in
        upper case.

    :param df: a Dataframe similar to create_cpg_input_df()
    :returns: a Dataframe similar to create_cpg_input_df() with a data-problem

    """
    batches = df["batch"].unique()
    return df.assign(
        item=lambda d: np.where(
            d["batch"].isin(np.random.choice(batches)),
            d["item"].str.upper(),
            d["item"],
        )
    )


def problem2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a simulated problem (2) with the input data.

    Problem: missing data for a minor segment (location)

    :param df: a Dataframe similar to create_cpg_input_df()
    :returns: a Dataframe similar to create_cpg_input_df() with a data-problem

    """
    batches = df["batch"].unique()
    return df.loc[
        lambda d: ~(
            d["batch"].isin(np.random.choice(batches) & (d["location"] == "BE"))
        )
    ]


def problem3(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a simulated problem (3) with the input data.

    Problem: duplicates because of a double loaded batch

    :param df: a Dataframe similar to create_cpg_input_df()
    :returns: a Dataframe similar to create_cpg_input_df() with a data-problem

    """
    batches = df["batch"].unique()
    return df.assign(
        batch=lambda d: np.where(
            d["batch"].isin(np.random.choice(batches)), batches.max() + 7, d["batch"],
        )
    )


def problem4(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a simulated problem (4) with the input data.

    Problem: duplicates because of a double loaded batch, with unique index

    :param df: a Dataframe similar to create_cpg_input_df()
    :returns: a Dataframe similar to create_cpg_input_df() with a data-problem

    """
    return problem3(df).reset_index().assign(line_id=lambda d: d.index)


def aggregate(
    df: pd.DataFrame,
    groupby: Optional[List[str]] = None,
    dims: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Aggregate data Suburbia CPG data in long format for plotting in seaborn.

    Aggregate data in so called long format for plotting.

    :param df: a Dataframe similar to create_cpg_input_df() or
    :param groupby: a list with one or more of the following columns or
        create_cpg_df() with at least column date "date":
        "date", "batch", "data_sources", "location","category", "brand"
    :param dims: a list with one ore more of the following dimensions to show:
        "rows", "quantity", "volume_eur"
    :returns: a Dataframe ready for plotting in seaborn longform format for
        relplot. dimensions:
            rows: counts the number of rows.
            quantity: the sum of all items sold.
            volume_eur: the sales volume in euro.
    """
    groupby = groupby or ["date"]
    dims = dims or ["rows"]
    return (
        df.groupby(groupby)
        .agg(
            rows=("date", "count"),
            volume_eur=("volume_eur", "sum"),
            quantity=("quantity", "sum"),
        )
        .reset_index()
        .melt(id_vars=groupby, value_vars=dims, var_name="dim",)
    )


def plot_data(
    df: pd.DataFrame,
    groupby: Optional[List[str]] = None,
    dims: Optional[List[str]] = None,
) -> None:
    """
    Plot the requested data from a suburbia cleaned or input DataFrame.

    :param df: a Dataframe similar to create_cpg_input_df() or
    :param groupby: a list with one or more of the following columns or
        create_cpg_df() with at least column date "date":
        "date", "batch", "data_sources", "location","category", "brand"
    :param dims: a list with one ore more of the following dimensions to show:
            rows: counts the number of rows.
            quantity: the sum of all items sold.
            volume_eur: the sales volume in euro.
    """
    groupby = groupby or ["date"]
    dims = dims or ["rows", "volume_eur", "quantity"]

    if len(groupby) == 1:
        plot_settings = {"hue": "dim"}
    elif len(groupby) == 2:
        plot_settings = {"hue": groupby[1]}
    elif len(groupby) == 3:
        plot_settings = {"hue": groupby[1], "row": groupby[2]}
    else:
        raise ValueError

    plot_settings.update(
        {
            "x": groupby[0],
            "y": "value",
            "col": "dim",
            "kind": "line",
            "legend": "full",
            "estimator": None,
            "facet_kws": {"sharey": False},
        }
    )

    sns.relplot(**plot_settings, data=aggregate(df, groupby, dims))
    plt.show()
