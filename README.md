# Analytics tools for CPG datasets

This repository incorporates several analytics tools used during processing
of CPG datasets at Suburbia. They are separated into three different categories:

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

For each time series, we provide an indication of correctness,
a chart of actual values together with a predicted interval, and a summary
of problematic dates.

### Unbranded tagger
An algorithm for tagging new line items as "unbranded" given a list of items
that have already been tagged as "unbranded".

---
#### Usage
Install:

```bash
$ pip install .
```

For an example IPython notebook and files, go to `example/`.

To start stability settings app, run:
```bash
$ R -f R/stability_settings/app.R
```
