import pandas as pd
import numpy as np

from time import time

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.style.use('seaborn-white')
import seaborn as sns

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
from plotly import tools

from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf

from IPython.display import Markdown, display
import warnings

def get_model_performance(X_test, y_true, y_pred, probs):
    """
    Extract basic machine learning model performance.
    """
    
    accuracy = accuracy_score(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    preds = probs[:, 1]
    fpr, tpr, threshold = roc_curve(y_true, preds)
    roc_auc = auc(fpr, tpr)

    return accuracy, matrix, f1, fpr, tpr, roc_auc

def plot_model_performance(model, X_test, y_true):
    """
    Plot ROC Curve.
    """
    
    y_pred = model.predict(X_test)
    probs = model.predict_proba(X_test)
    accuracy, matrix, f1, fpr, tpr, roc_auc = get_model_performance(X_test, y_true, y_pred, probs)

    display(Markdown('#### Accuracy of the model :'))
    print(accuracy)
    display(Markdown('#### F1 score of the model :'))
    print(f1)

    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot(1, 2, 1)
    sns.heatmap(matrix, annot=True, cmap='Blues', fmt='g')
    plt.title('Confusion Matrix')

    ax = fig.add_subplot(1, 2, 2)
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic curve')
    plt.legend(loc="lower right")


def add_to_df_algo_metrics(algo_metrics, model, fair_metrics, preds, probs, name):
    """
    Add fairness metrics to dataframe of metrics.
    
    Parameters:
    algo_metrics (model metrics): Original metrics dataframe
    model (sklearn model): Model trained on the data
    fair_metrics (array): Fairness metrics
    preds (array): Model predictions
    probs (array): Prediction probabilities
    name (string): Label of metric/algorithm to be measure
    
    Returns:
    Method to append algorithm metrics.
    """
    
    return algo_metrics.append(pd.DataFrame(data=[[model, fair_metrics, preds, probs]], columns=['model', 'fair_metrics', 'prediction', 'probs'], index=[name]))

def get_fair_metrics(dataset, pred, pred_is_dataset=False):
    """
    Measure fairness metrics.
    
    Parameters: 
    dataset (pandas dataframe): Dataset
    pred (array): Model predictions
    pred_is_dataset, optional (bool): True if prediction is already part of the dataset, column name 'labels'.
    
    Returns:
    fair_metrics: Fairness metrics.
    """
    if pred_is_dataset:
        dataset_pred = pred
    else:
        dataset_pred = dataset.copy()
        dataset_pred.labels = pred
    
    cols = ['statistical_parity_difference', 'equal_opportunity_difference', 'average_abs_odds_difference',  'disparate_impact', 'theil_index']
    obj_fairness = [[0,0,0,1,0]]
    
    fair_metrics = pd.DataFrame(data=obj_fairness, index=['objective'], columns=cols)
    
    for attr in dataset_pred.protected_attribute_names:
        idx = dataset_pred.protected_attribute_names.index(attr)
        privileged_groups =  [{attr:dataset_pred.privileged_protected_attributes[idx][0]}] 
        unprivileged_groups = [{attr:dataset_pred.unprivileged_protected_attributes[idx][0]}] 
        
        classified_metric = ClassificationMetric(dataset, 
                                                     dataset_pred,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)

        metric_pred = BinaryLabelDatasetMetric(dataset_pred,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)

        acc = classified_metric.accuracy()

        row = pd.DataFrame([[metric_pred.mean_difference(),
                                classified_metric.equal_opportunity_difference(),
                                classified_metric.average_abs_odds_difference(),
                                metric_pred.disparate_impact(),
                                classified_metric.theil_index()]],
                           columns  = cols,
                           index = [attr]
                          )
        fair_metrics = fair_metrics.append(row)    
    
    fair_metrics = fair_metrics.replace([-np.inf, np.inf], 2)
        
    return fair_metrics

def plot_fair_metrics(fair_metrics):
    """
    Plots the fairness metrics.
    """
    fig, ax = plt.subplots(figsize=(20,4), ncols=5, nrows=1)

    plt.subplots_adjust(
        left    =  0.125, 
        bottom  =  0.1, 
        right   =  0.9, 
        top     =  0.9, 
        wspace  =  .5, 
        hspace  =  1.1
    )

    y_title_margin = 1.2

    plt.suptitle("Fairness metrics", y = 1.09, fontsize=20)
    sns.set(style="dark")

    cols = fair_metrics.columns.values
    obj = fair_metrics.loc['objective']
    size_rect = [0.2,0.2,0.2,0.4,0.25]
    rect = [-0.1,-0.1,-0.1,0.8,0]
    bottom = [-1,-1,-1,0,0]
    top = [1,1,1,2,1]
    bound = [[-0.1,0.1],[-0.1,0.1],[-0.1,0.1],[0.8,1.2],[0,0.25]]

    display(Markdown("### Check bias metrics :"))
    display(Markdown("A model can be considered bias if just one of these five metrics show that this model is biased."))
    for attr in fair_metrics.index[1:len(fair_metrics)].values:
        display(Markdown("#### For the %s attribute :"%attr))
        check = [bound[i][0] < fair_metrics.loc[attr][i] < bound[i][1] for i in range(0,5)]
        display(Markdown("With default thresholds, bias against unprivileged group detected in **%d** out of 5 metrics"%(5 - sum(check))))
        

    for i in range(0,5):
        plt.subplot(1, 5, i+1)
        ax = sns.barplot(x=fair_metrics.index[1:len(fair_metrics)], y=fair_metrics.iloc[1:len(fair_metrics)][cols[i]])
        
        for j in range(0,len(fair_metrics)-1):
            a, val = ax.patches[j], fair_metrics.iloc[j+1][cols[i]]
            marg = -0.2 if val < 0 else 0.1
            ax.text(a.get_x()+a.get_width()/5, a.get_y()+a.get_height()+marg, round(val, 3), fontsize=15,color='black')

        plt.ylim(bottom[i], top[i])
        plt.setp(ax.patches, linewidth=0)
        ax.add_patch(patches.Rectangle((-5,rect[i]), 10, size_rect[i], alpha=0.3, facecolor="green", linewidth=1, linestyle='solid'))
        plt.axhline(obj[i], color='black', alpha=0.3)
        plt.title(cols[i])
        ax.set_ylabel('')    
        ax.set_xlabel('')
        
def get_fair_metrics_and_plot(data, model, plot=True, model_aif=False):
    """
        Computes fairness metrics and plots them.
    """
    pred = model.predict(data).labels if model_aif else model.predict(data.features)
    
    fair = get_fair_metrics(data, pred)

    if plot:
        
        # The visualisation of this function is inspired by the dashboard on the demo of IBM aif360 
        plot_fair_metrics(fair)
        display(fair)
    
    return fair

def prepare_data(data, priv_category, priv_value, target_label, priv_target_value, ignore_cols=None):
    """
    Prepare dataset for bias mitigation.
    
    Paramters:
    data (pandas dataframe): Data to fix (for fairness)
    priv_category (string): Column name that contains the privileged value (e.g. Race, Gender, etc)
    priv_value (string): Value or type in the column that denotes the privileged attribute (e.g. White, Male, etc)
    target_label (string): Column name of target variable (e.g. income, loan score, etc)
    priv_target_value (string): Value in target that favors the privileged (e.g. High income, favorable loan score, credit acceptance, etc).
                                Must be boolean (so if target is numeric, convert to categorical by thresholding before processing.)
    ignore_cols, optional (list of string): List of columns to exclude from bias assessment and modeling.
    
    Returns:
    data_priv (standard Dataset): Dataset prepared by aif360 for processing
    encoders (dict): dictionary of encoding models
    numerical_features (list): List of numerical columns
    categorical_features (list) List of categorical columns
    """
    
    if ignore_cols:
        data = data.drop(ignore_cols, axis=1)
    else:
        pass
    
    # Get categorical features
    categorical_features = data.columns[data.dtypes == 'object']
    data_encoded = data.copy()
    
    # Store categorical names and encoders
    categorical_names = {}
    encoders = {}

    # Use Label Encoder for categorical columns (including target column)
    for feature in categorical_features:
        le = LabelEncoder()
        le.fit(data_encoded[feature])

        data_encoded[feature] = le.transform(data_encoded[feature])

        categorical_names[feature] = le.classes_
        encoders[feature] = le
        
    # Scale numeric columns
    numerical_features = [c for c in data.columns.values if c not in categorical_features]

    for feature in numerical_features:
        val = data_encoded[feature].values[:, np.newaxis]
        mms = MinMaxScaler().fit(val)
        data_encoded[feature] = mms.transform(val)
        encoders[feature] = mms

    data_encoded = data_encoded.astype(float)
    
    privileged_class = np.where(categorical_names[priv_category]==priv_value)[0]
    encoded_target_label = np.where(categorical_names[target_label]==priv_target_value)[0]
    
    data_priv = StandardDataset(data_encoded, 
                               label_name=target_label, 
                               favorable_classes=encoded_target_label, 
                               protected_attribute_names=[priv_category], 
                               privileged_classes=[privileged_class])
    
    return data_priv, encoders, numerical_features, categorical_features

def show_bias(data, priv_category, priv_value, target_label, unencoded_target_label, cols_to_drop):
    """
    Show biases in the data.
    
    Parameters:
    data (pandas dataframe): Data to fix (for fairness)
    priv_category (string): Column name that contains the privileged value (e.g. Race, Gender, etc)
    priv_value (string): Value or type in the column that denotes the privileged attribute (e.g. White, Male, etc)
    target_label (string): Column name of target variable (e.g. income, loan score, etc)
    priv_target_value (string): Value in target that favors the privileged (e.g. High income, favorable loan score, credit acceptance, etc).
                                Must be boolean (so if target is numeric, convert to categorical by thresholding before processing.)
    ignore_cols, optional (list of string): List of columns to exclude from bias assessment and modeling.
    
    Returns:
    Bias analysis chart.
    """
    data_orig, encoders, numerical_features, categorical_features = prepare_data(data, priv_category, priv_value, target_label, 
                                 priv_target_value=unencoded_target_label, ignore_cols=cols_to_drop)

    data_orig_train, data_orig_test = data_orig.split([0.7], shuffle=True)
    
    # Train and save the models
    rf_orig = RandomForestClassifier().fit(data_orig_train.features, 
                         data_orig_train.labels.ravel(), 
                         sample_weight=data_orig_train.instance_weights)

    
    fair = get_fair_metrics_and_plot(data_orig_test, rf_orig)

def compare_fair_metrics(algo_metrics, priv_category):
    """
    Shows a table of all the bias metrics of all mitigations.
    
    Parameters: 
    algo_metrics: dataframe of all metrics and 
    attr: privileged category
    
    Returns:
    df_metrics: a dataframe 
    """
    
    df_metrics = pd.DataFrame(columns=algo_metrics.loc['Origin','fair_metrics'].columns.values)
    for fair in algo_metrics.loc[:,'fair_metrics']:
        df_metrics = df_metrics.append(fair.loc[priv_category], ignore_index=True)

    df_metrics.index = algo_metrics.index.values
    df_metrics = df_metrics.replace([np.inf, -np.inf], np.NaN)
    
    return df_metrics
