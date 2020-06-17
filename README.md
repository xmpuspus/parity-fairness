# Parity

This repository contains codes that demonstrate the use of fairness metrics, bias mitigations and explainability tool.

### Installation

In order for the explainability modules to work, first you have to install shap through `conda` like so:

```console
foo@bar:~$ conda install -c conda-forge shap
```

Install using:

```console
foo@bar:~$ pip install parity-fairness
```

### Bias Measurement Usage

Setup the data such that the target column is a binary string target. Then find out which features are the `privileged categories` and which values are `privileged values`. Afterwards, feed them into the function called `show_bias` like:

```
from parity.fairness_metrics import show_bias

priv_category = 'Race-White'
priv_value = 'True'
target_label = 'high pay'
unencoded_target_label = 'True'
cols_to_drop = ''

show_bias(data, priv_category, priv_value, target_label, unencoded_target_label, cols_to_drop)
```

# Bias and Fairness

A common problem with most machine learning models is bias from data. This notebook shows how to measure those biases and perform bias mitigation. A python package called [aif360](https://github.com/IBM/AIF360) can give us metrics and algorithms for bias measurement and mitigation

### Metrics

* Statistical Parity Difference
* Equal Opportunity Difference
* Average Absolute Odds Difference
* Disparate Impact
* Theil Index

Some metrics need predictions while others just the original dataset. This is why we will use 2 classes of the aif360 package : `ClassificationMetric` and `BinaryLabelDatasetMetric`. 

### For metrics that require predictions: 
* [Equal Opportunity Difference: ](https://aif360.readthedocs.io/en/latest/modules/metrics.html#aif360.metrics.ClassificationMetric.equal_opportunity_difference)  `equal_opportunity_difference()`
* [Average Absolute Odds Difference: ](https://aif360.readthedocs.io/en/latest/modules/metrics.html#aif360.metrics.ClassificationMetric.average_abs_odds_difference)  `average_abs_odds_difference()`
* [Theil Index : ](https://aif360.readthedocs.io/en/latest/modules/metrics.html#aif360.metrics.ClassificationMetric.theil_index) `theil_index()`

### For metrics that don't require predictions: 
* [Statistical Parity Difference: ](https://aif360.readthedocs.io/en/latest/modules/metrics.html#aif360.metrics.BinaryLabelDatasetMetric.statistical_parity_difference)  `statistical_parity_difference()`
* [Disparate Impact: ](https://aif360.readthedocs.io/en/latest/modules/metrics.html#aif360.metrics.ClassificationMetric.disparate_impact)  `disparate_impact()`

