import pandas as pd

from fairlearn.reductions import GridSearch
from fairlearn.reductions import ExponentiatedGradient
from fairlearn.reductions import DemographicParity, ErrorRate
from fairlearn.widget import FairlearnDashboard

def gridSearch(model, X_train, Y_train, A_train, grid_size):
    """
    Generates a sequence of relabellings and reweightings, and trains a predictor for each. 
    Only applicable for binary feature.
    
    Parameters:
    x_train: input data for training model
    y_train: list of ground truths
    model: the unmitigated algorthmic model
    
    Returns a dataframe of the different predictors and its accuracy scores and disparity scores.
    
    """
    sweep = GridSearch(model, constraints=DemographicParity(), grid_size=grid_size)

    # we extract the full set of predictors from the `GridSearch` object
    sweep.fit(X_train, Y_train, sensitive_features=A_train)
        
    predictors = sweep._predictors
    
    """
    Remove the predictors which are dominated in the error-disparity space by others from the sweep 
    (note that the disparity will only be calculated for the protected attribute; 
    other potentially protected attributes will not be mitigated)
   
    In general, one might not want to do this, since there may be other considerations beyond the strict 
    optimisation of error and disparity (of the given protected attribute).
    """
    errors, disparities = [], []
    for m in predictors:
        classifier = lambda X: m.predict(X)
    
        error = ErrorRate()
        error.load_data(X_train, pd.Series(Y_train), sensitive_features=A_train)
        disparity = DemographicParity()
        disparity.load_data(X_train, pd.Series(Y_train), sensitive_features=A_train)
    
        errors.append(error.gamma(classifier)[0])
        disparities.append(disparity.gamma(classifier).max())
    
    all_results = pd.DataFrame( {"predictor": predictors, "error": errors, "disparity": disparities})

    non_dominated = []
    for row in all_results.itertuples():
        errors_for_lower_or_eq_disparity = all_results["error"][all_results["disparity"]<=row.disparity]
        if row.error <= errors_for_lower_or_eq_disparity.min():
            non_dominated.append(row.predictor)
            
    return non_dominated

def show_comparison(model, X_test, Y_test, A_test, protected_features, non_dominated):
    
    """
    Returns Dashboard to show comparison of models based on the trade off of the disparity and accuracy
    
    """
    dashboard_predicted = {"unmitigated": model.predict(X_test)}
    for i in range(len(non_dominated)):
        key = "dominant_model_{0}".format(i)
        value = non_dominated[i].predict(X_test)
        dashboard_predicted[key] = value
   
    dashboard = FairlearnDashboard(sensitive_features=A_test, sensitive_feature_names=protected_features,
                   y_true=Y_test,
                   y_pred=dashboard_predicted)
    
    return dashboard
