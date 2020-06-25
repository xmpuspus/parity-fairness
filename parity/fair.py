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
from aif360.algorithms.preprocessing import LFR, Reweighing, DisparateImpactRemover
from aif360.algorithms.inprocessing import AdversarialDebiasing, PrejudiceRemover
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing, EqOddsPostprocessing, RejectOptionClassification
from fairlearn.widget import FairlearnDashboard

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf

from IPython.display import Markdown, display
import warnings



def fairness_dashboard(A_test, protected_features, y_true, y_pred):
    """
    Displays an interactive dashboards that shows fairness metrics given the data, 
    protected features and the train-test targets.
    
    Parameters:
    A_test (pandas dataframe): Input data to be analyzed for fairness metrics.
    protected_features (list): Features or columns we need to account for potential biases.
    y_true (list): List of ground truths
    y_pred (list): List of predictions
    
    Returns:
    Interactive dashboard for fairness metrics.
    """
    
    FairlearnDashboard(sensitive_features=A_test,
                   sensitive_feature_names=protected_features,
                   y_true=y_true.values.tolist(),
                   y_pred=[y_pred.tolist()])
    
    
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


def decode_dataset(df, encoders, numerical_features, categorical_features):
    """
    Decodes output dataset based on previous preprocessing encoders:
    
    Parameters:
    df (pandas dataframe): Data to be decoded
    encoders (dict): Dictionary of encoding models
    numerical_features (list): List of numerical columns
    categorical_features (list) List of categorical columns
    
    Returns:
    df (pandas dataframe): Decoded dataframe
    """
    
    for feat in df.columns.values:
        if feat in numerical_features:
            df[feat] = encoders[feat].inverse_transform(np.array(df[feat]).reshape(-1, 1))
    for feat in categorical_features:
        df[feat] = encoders[feat].inverse_transform(df[feat].astype(int))
    return df

def get_attributes(data, selected_attr=None):
    """
    Get privileged groups from structured data.
    
    Parameters:
    data (aif360.datasets.standard_dataset.StandardDataset): Structured dataset.
    selected_attr, optional (string): 
    """
    
    unprivileged_groups = []
    privileged_groups = []
    if selected_attr == None:
        selected_attr = data.protected_attribute_names
    
    for attr in selected_attr:
            idx = data.protected_attribute_names.index(attr)
            privileged_groups.append({attr:data.privileged_protected_attributes[idx]}) 
            unprivileged_groups.append({attr:data.unprivileged_protected_attributes[idx]}) 

    return privileged_groups, unprivileged_groups


def convert_to_pd_dataframe(structured_data):
    """
    Converts AIF360 structured dataset to pandas dataframe.
    
    Parameters:
    structured_data (aif360.datasets.standard_dataset.StandardDataset): Structured dataset.
    
    Returns:
    df (pandas dataframe): Pandas dataframe.
    """
    
    df = structured_data.convert_to_dataframe()[0]
    return df


def disparate_impact_remover(structured_data):
    """
    Perform disparate impact removal from dataset and convert to pandas dataframe.
    
    Parameters:
    aif_standard_data (aif360.datasets.standard_dataset.StandardDataset): Structured dataset.
    
    Returns:
    data_transf_df (pandas dataframe): Pandas dataframe.
    """
    
    DIR = DisparateImpactRemover()
    data_transf = DIR.fit_transform(structured_data)
    data_transf_df = convert_to_pd_dataframe(data_transf)
    
    return data_transf_df

def learning_fair_representation(structured_data, priv_category):
    """
    Remove bias from dataset using LFR.
    Parameters:
    structured_data (aif360.datasets.standard_dataset.StandardDataset): Structured dataset.
    priv_category (string): Column with privileged class.
    Returns:
    data_transf_df (pandas dataframe): Pandas dataframe.
    """
    
    # Get privileged and unprivileged groups
    privileged_groups, unprivileged_groups = get_attributes(structured_data, selected_attr=[priv_category])
    LFR_model = LFR(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups, k=1, verbose=0)
    
    # Remove bias
    data_transf = LFR_model.fit_transform(structured_data)
    
    # Convert to pandas dataframe
    data_transf_df = convert_to_pd_dataframe(data_transf)
    
    return data_transf_df

def reweight(structured_data, priv_category):
    """
    Remove bias from dataset using Reweighing.
    Parameters:
    structured_data (aif360.datasets.standard_dataset.StandardDataset): Structured dataset.
    priv_category (string): Column with privileged class.
    Returns:
    data_transf_df (pandas dataframe): Pandas dataframe.
    """
    
    # Get privileged and unprivileged groups
    privileged_groups, unprivileged_groups = get_attributes(structured_data, selected_attr=[priv_category])
    RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    
    # Remove bias
    data_transf = RW.fit_transform(structured_data)
    
    # Convert to pandas dataframe
    data_transf_df = convert_to_pd_dataframe(data_transf)
    
    return data_transf_df

def structured_data_train_test_split(structured_data, train_size=0.3):
    """
    Train-test-split the AIF360 structured data.
    
    Parameters:
    structured_data (aif360.datasets.standard_dataset.StandardDataset): Structured dataset.
    train_size (float): Training size (from 0 t0 1).
    
    Returns:
    structured_data_train (aif360.datasets.standard_dataset.StandardDataset): Structured train dataset.
    structured_data_test (aif360.datasets.standard_dataset.StandardDataset): Structured test dataset.
    """
    
    structured_data_train, structured_data_test = structured_data.split([train_size], shuffle=True)
    
    return structured_data_train, structured_data_test

def adversarial_debias(structured_data_train, structured_data_test, priv_category, scope_name='debiased_classifier', num_epochs=10):
    """
    Remove bias from dataset using Adversarial debiasing. Must use Tensorflow version 1.14.
    
    Parameters:
    structured_data (aif360.datasets.standard_dataset.StandardDataset): Structured dataset.
    priv_category (string): Column with privileged class.
    scope_name (string): Name of scope to debias. See documentation on adversarial debiasing for details.
    num_epochs (int): Epochs to train with.
    Returns:
    data_transf_train_df (pandas dataframe): Pandas dataframe of train set
    data_transf_df (pandas dataframe): Pandas dataframe of test set
    """
    
    sess = tf.Session()
    
    # Get privileged and unprivileged groups
    privileged_groups, unprivileged_groups = get_attributes(structured_data_train, selected_attr=[priv_category])
    debiased_model = AdversarialDebiasing(privileged_groups = privileged_groups,
                          unprivileged_groups = unprivileged_groups,
                          scope_name=scope_name,
                          num_epochs=num_epochs,
                          debias=True, sess=sess)
    
    debiased_model.fit(structured_data_train)
    
    # Remove bias
    data_train_pred = debiased_model.predict(structured_data_train)
    data_pred = debiased_model.predict(structured_data_test)
    
    # Convert to pandas dataframe
    data_transf_train_df = convert_to_pd_dataframe(data_train_pred)
    data_transf_test_df = convert_to_pd_dataframe(data_pred)
    
    return data_transf_train_df, data_transf_test_df

def prejudice_remover(structured_data_train, structured_data_test, priv_category, eta=25):
    """
    Remove bias from dataset using Prejudice Remover Regularizer.
    
    Parameters:
    structured_data (aif360.datasets.standard_dataset.StandardDataset): Structured dataset.
    priv_category (string): Column with privileged class.
    eta (int): Regularization parameter
    Returns:
    data_transf_train_df (pandas dataframe): Pandas dataframe.
    data_transf_test_df (pandas dataframe): Pandas dataframe.
    """
    
    debiased_model = PrejudiceRemover(sensitive_attr=priv_category, eta=eta)
    debiased_model.fit(structured_data_train)
    
    # Remove bias
    data_train_pred = debiased_model.predict(structured_data_train)
    data_pred = debiased_model.predict(structured_data_test)
    
    # Convert to pandas dataframe
    data_transf_train_df = convert_to_pd_dataframe(data_train_pred)
    data_transf_test_df = convert_to_pd_dataframe(data_pred)
    
    return data_transf_train_df, data_transf_test_df

def calibrate_equality_of_odds(train_data, test_data, predicted_data, priv_category, cost_constraint="fnr"):
    """
    Remove bias from dataset using Calibrated Equality of Odds.
    
    Parameters:
    train_data (aif360.datasets.standard_dataset.StandardDataset): Structured dataset for train data.
    test_data (aif360.datasets.standard_dataset.StandardDataset): Structured dataset for test data.
    predicted_data (aif360.datasets.standard_dataset.StandardDataset): Structured dataset for predicted data.
    priv_category (string): Column with privileged class.
    cost_constraint (str): ("fnr" or false negative rate, "fpr" or false positive rate, and "weighted" for custom weights)
    
    Returns:
    data_transf_pred: aif360's standard dataset
    CPP: algorithmic model for mitigation
    """
    
    privileged_groups, unprivileged_groups = get_attributes(train_data, selected_attr=[priv_category])
    CPP = CalibratedEqOddsPostprocessing(privileged_groups = privileged_groups,
                                     unprivileged_groups = unprivileged_groups,
                                     cost_constraint=cost_constraint,
                                     seed=42)

    CPP = CPP.fit(test_data, predicted_data)
    data_transf_pred = CPP.predict(predicted_data)
       
    return data_transf_pred, CPP

def reject_option(train_data, test_data, predicted_data, priv_category, cost_constraint="fnr"):
    """
    Remove bias from dataset using Reject Option.
    
    Parameters:
    train_data (aif360.datasets.standard_dataset.StandardDataset): Structured dataset for train data.
    test_data (aif360.datasets.standard_dataset.StandardDataset): Structured dataset for test data.
    predicted_data (aif360.datasets.standard_dataset.StandardDataset): Structured dataset for predicted data.
    priv_category (string): Column with privileged class.
    cost_constraint (str): ("fnr" or false negative rate, "fpr" or false positive rate, and "weighted" for custom weights)
    
    Returns:
    data_transf_pred: aif360's standard dataset
    ROC: algorithmic model for mitigation
    """
    privileged_groups, unprivileged_groups = get_attributes(train_data, selected_attr=[priv_category])
    ROC = RejectOptionClassification(privileged_groups = privileged_groups,
                             unprivileged_groups = unprivileged_groups)
    
    ROC = ROC.fit(test_data, predicted_data)
    data_transf_pred = ROC.predict(predicted_data)
       
    return data_transf_pred, ROC
