# Analytics tools for CPG datasets

This repository incorporates several analytics tools used during processing
of CPG (consumer packaged goods) datasets at Suburbia as well as notebooks
demonstrating their usage at Suburbia.
They are separated into four different categories:

### Merchant stability
Set of functions for classification of stable merchants based on four criteria:
* lifespan in days,
* relative activity (percentage of days with sales activity),
* volatility of sales quantities,
* longest gap (longest period without activity).

Stability of a merchant is determined by comparing these criteria with a set
of thresholds.

Thresholds can be chosen using an associated _Stability settings_ Shiny app.

### Outlier detection
Tools for detection of outliers in a set of specified metrics,
based on [Prophet](https://facebook.github.io/prophet/).

We fit a time series model for each metric we are interested in, using data
up until a reference date. All values collected after the reference date will
be then compared with predicted values. If observed value is outside of predicted
interval or is missing, the algorithm labels it as such.

For each time series, we provide an indication of correctness,
a chart of actual values together with a predicted interval, and a summary
of problematic dates.

Next time the outlier detection is run, user can choose to exclude certain
problematic dates to prevent the model from training on this data.

### Unbranded tagger
An algorithm for tagging new line items as "unbranded" given a list of items
that have already been tagged as "unbranded".

### Data simulation
Tools for simulation of data for demonstration of data processing at Suburbia,
available through example Jupyter notebook `notebooks/Data-Science at Suburbia.ipynb`.

---
#### Usage
Install:

```bash
$ pip install .
```

For an IPython notebooks and example files, go to `notebooks/`.

To start Stability settings app, run:
```bash
$ R -f R/stability_settings/app.R
```
