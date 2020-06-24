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

A common problem with most machine learning models is bias from data. This notebook shows how to measure those biases and perform bias mitigation. A python package called [aif360](https://github.com/IBM/AIF360) can give us metrics and algorithms for bias measurement and mitigation.

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

### People + AI Research

A project by Google, this module helps visualize inference results, visualize feature attributions, arrange datapoints by similarity, edit a datapoint and see how it performs, compare counterfactuals to datapoints, and test algorithmic fairness constraints. More details can be found [here](https://pair-code.github.io/what-if-tool/).

To use, simply add:

```
from parity import pair
```

Sample code to show the GUI:

```
from parity.pair import *

### Insert more code here

# Load up the test dataset
test_csv_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
test_df = pd.read_csv(test_csv_path, names=csv_columns, skipinitialspace=True,
  skiprows=1)

num_datapoints = 2000
tool_height_in_px = 1000
make_label_column_numeric(test_df, label_column, lambda val: val == '>50K.')
test_examples = df_to_examples(test_df[0:num_datapoints])
labels = ['Under 50K', 'Over 50K']

# Setup the tool with the test examples and the trained classifier
config_builder = config(test_examples, classifier, labels, feature_spec)

## Visualization only appears through jupyter notebook (not jupyter lab)
WitWidget(config_builder, height=tool_height_in_px)
```

### Bias Mitigation

There are three types of bias mitigation:
- Pre-processing algorithms : they are used before training the model
- In-processing algorithms : they are fair classifiers so it's during the training
- Post-processing algorithms : they are used after training the model

### Bias Mitigation Usage

Before using the different bias mitigation functions, necessary objects must be prepared using the function `prepare_data`. Once the dataset is processed which in this case is stored in `data_orig`, split this into testing and training datasets. Along with other different objects: `encoders`, `numerical_features` and `categorical_features` these will be fed appropriately when using any mitigation tools.

```
from parity.fairness_metrics import prepare_data

priv_category = 'Race-White'
priv_value = 'True'
target_label = 'high pay'
unencoded_target_label = 'True'
cols_to_drop = ''

data_orig, encoders, numerical_features, categorical_features = prepare_data(data, priv_category, priv_value, target_label, 
                             priv_target_value=unencoded_target_label, ignore_cols=cols_to_drop)
                             
data_orig_train, data_orig_test = data_orig.split([0.7], shuffle=True)
```

#### Preprocess Mitigations

Using the prepared data and identifying which `priveleged categoery`, bias is removed using any of the preprocessing mitigation function tools. Afterwards, this must be decoded into pandas dataframe for readability by feeding the transformed data and other objects into the `decode_dataset` function.

* [Disparate Impact Remover: ](https://dl.acm.org/doi/10.1145/2783258.2783311) `disparate_impact_remover()`

```
from parity.fair import disparate_impact_remover, decode_dataset

data_transf_df = disparate_impact_remover(data_orig_train)

decoded_df = decode_dataset(data_transf_df, encoders, numerical_features, categorical_features)
```

* [Learning Fair Representation:](http://proceedings.mlr.press/v28/zemel13.html) `learning_fair_representation()`

```
from parity.fair import learning_fair_representation, decode_dataset

data_transf_df = learning_fair_representation(data_orig_train, priv_category)

decoded_df = decode_dataset(data_transf_df, encoders, numerical_features, categorical_features)
```

* [Reweighing:](https://link.springer.com/article/10.1007%2Fs10115-011-0463-8) `reweight()`

```
from parity.fair import reweight, decode_dataset

data_transf_df = reweight(data_orig_train, priv_category)

decoded_df = decode_dataset(data_transf_df, encoders, numerical_features, categorical_features)
```

#### Inprocess Mitigations

For inprocessing, the `structured_data_train_test_split` is used to split the prepared original dataset. These are then fed on a chosen mitigation tool and must be decoded into pandas dataframe for readability using `decode_dataset` function.

```
data_orig_train, data_orig_test = structured_data_train_test_split(data_orig)
```

* [Adversarial Debiasing:](https://arxiv.org/pdf/1801.07593.pdf)`adversarial_debias()`

```
from parity.fair import adversarial_debias, decode_dataset

data_transf_train_df, data_transf_test_df = adversarial_debias(data_orig_train, data_orig_test, priv_category)

decoded_train_df = decode_dataset(data_transf_train_df, encoders, numerical_features, categorical_features)

decoded_test_df = decode_dataset(data_transf_test_df, encoders, numerical_features, categorical_features)
```

* [Prejudice Remover Regularizer:](https://rd.springer.com/chapter/10.1007/978-3-642-33486-3_3)`prejudice_remover()`

```
from parity.fair import prejudice_remover, decode_dataset

data_transf_train_df, data_transf_test_df = prejudice_remover(data_orig_train, data_orig_test, priv_category, eta=25)

decoded_train_df = decode_dataset(data_transf_train_df, encoders, numerical_features, categorical_features)

decoded_test_df = decode_dataset(data_transf_test_df, encoders, numerical_features, categorical_features)
```

* [Grid Search:](https://arxiv.org/abs/1803.02453)`gridSearch()`

This preprocessing tool has a different approch of debiasing. It works by generating a sequence of relabellings and reweighings, and trains a predictor for each. These values are then fed in the function called `show_comparison` to create a dashboard that shows the comparison of before and after the mitigation.

```
from parity.reduction import gridSearch

values = gridSearch(model = unmitigated_model, X_train = X_train, Y_train = Y_train, A_train = A_train, grid_size = 71)

show_comparison(model = unmitigated_model, X_test = X_test, Y_test = Y_test,
                          A_test = A_test, protected_features =['sex'], non_dominated = values)                        
```
    
#### Postprocess Mitigations

This type of mitigation needs the `prediction scores` from using the original model. Along with this, prepared testing and training datasets are fed into the mitigation function tool. Afterwards, the newly mitigated predictions must be decoded into a pandas dataframe for readability using `decode_dataset` function.

* [Calibrated Equality of Odds](https://papers.nips.cc/paper/7151-on-fairness-and-calibration)`calibrate_equality_of_odds()`

```
data_trans_pred = calibrate_equality_of_odds(data_orig_train, data_orig_test, data_orig_test_pred, priv_category)

decoded_pred_df = decode_dataset(data_trans_pred, encoders, numerical_features, categorical_features)
```

* [Reject Option Classification](https://ieeexplore.ieee.org/document/6413831) `reject_option`

```
data_trans_pred = reject_option(data_orig_train, data_orig_test, data_orig_test_pred, priv_category)

decoded_pred_df = decode_dataset(data_trans_pred, encoders, numerical_features, categorical_features)
```

### Fairness Dashboard

When creating a FairlearnDashboard one must provide the `input data` to be analyzed for fairness metrics, `sensitive attributes`, and `list of ground truths and prediction`. Once these are fed to the funciton called `fairness_dashboard`, a dashboard should displays the sensitive attributes followed by options of model performance metrics. After selecting, performance and prediction disparity must be shown.

```

from parity.fair import fairness_dashboard

sensitive_feature_names = ['Age', 'Sex']
A_test = data.iloc[X_test.index][sensitive_feature_names]

fairness_dashboard(A_test=A_test, protected_features=sensitive_feature_names, y_true=y_test, y_pred=y_pred)

```

# Explainability

Another common problem in machine learning is determining which features significantly explain the prediction of a model. The [SHAP or SHapley Additive exPlanations](https://github.com/slundberg/shap) library can be use to address this. To simply get scores of how features are useful at predicting a target variable, use `feature_importance` function. These are computed by providing the `model` and the `input train data`.

```

from parity.explainer import shap_feature_explainer, plot_prediction_causes, dependence_plots, feature_importances

features = feature_importances(model, X_train)

```

To compute SHAP values for each feature, feed the `model` and `input train data` in the `shap_feature_explainer` function. 

```

shap_values = shap_feature_explainer(model, X_train)

```

To plot and determine the prediction causes for each feature from model given a specific row or `index` in the data, use the `plot_prediction_causes` function.

```

plot_prediction_causes(model, X_train, shap_values, index=1)

```

On the otherhand, to plot the interaction effects of SHAP values of features with each other with respect to the target variable, use `dependence_plots` function. 

```

dependence_plots(X_train, shap_values)

```
