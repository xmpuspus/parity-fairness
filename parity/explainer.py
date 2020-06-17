import pandas as pd
import shap

def feature_importances(model, X):
    """
    Creates feature importance dataframe from model and data inputs.
    
    Parameters:
    model (pickled sklearn model): Trained model
    X (pandas dataframe): Input data used in training the model.
    
    Returns:
    feature_importance_df (pandas dataframe): Outputs dataframe of zipped feature names and feature importances
    """
    
    importances = model.feature_importances_
    feature_importance_df = (pd.DataFrame({'Feature': X.columns, 
                                           'Feature Importance': importances}).
                             set_index('Feature')
                             .sort_values('Feature Importance', ascending=False))
    
    return feature_importance_df


def shap_feature_explainer(model, X, plot=True):
    """
    Computes SHAP values for each feature from model and input data.
    
    Parameters:
    model (pickled sklearn model): Trained model
    X (pandas dataframe): Input data used in training the model.
    
    Returns: 
    shap_values (2D-array x number of target classes): SHAP values
    """
    
    explain = shap.TreeExplainer(model)
    shap_values = explain.shap_values(X)
    if plot:
        shap.summary_plot(shap_values, X, plot_type="bar", show=True,)
        
    return shap_values


def plot_prediction_causes(model, X, shap_values, index=1):
    """
    Computes prediction causes for each feature from model given a specific row in the data.
    
    Parameters:
    model (pickled sklearn model): Trained model
    X (pandas dataframe): Input data used in training the model.
    shap_values (2D-array x number of target classes): SHAP values
    index (int): index in the input data
    
    Returns: Plot of prediction causes.
    """
    
    explain = shap.TreeExplainer(model)
    expectation = explain.expected_value

    shap.force_plot(expectation[1], shap_values[1][index,:], 
                        X.iloc[index,:], matplotlib=True, show=True, figsize=(16, 5))
    
    
def dependence_plots(X, shap_values, rank=5):
    """
    Plots the interaction effects of SHAP values of features with each other wrt to the target variable.
    
    Parameters:
    
    X (pandas dataframe): Input data used in training the model.
    shap_values (2D-array x number of target classes): SHAP values.
    rank (int): Number of top ranking features to compare (descending shap values).
    
    Returns: Dependence plots of interaction effects.
    """
    
    ingest=('rank('+str(rank)+')')
    if len(shap_values) > 1:
        shap.dependence_plot(ingest, shap_values[-1], X, show=True)
    else:
        shap.dependence_plot(ingest, shap_values, X, show=False)
    