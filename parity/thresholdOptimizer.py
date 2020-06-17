# Fairlearn algorithms and utils
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.widget import FairlearnDashboard

def thresholdOptimizer(X_train, Y_train, A_train, model, constraint):
    """
    Parameters:
    y_train: input data for training the model
    X_train: list of ground truths
    constraints: either "demographic_parity" or "equalized_odds"
    
    Returns the predictions of the optimized model
    """
    postprocess_est = ThresholdOptimizer(estimator=model,constraints=constraint)
    
    # Balanced data set is obtained by sampling the same number of points from the majority class (Y=0)
    # as there are points in the minority class (Y=1)
    
    Y_train = pd.Series(Y_train)
    balanced_idx1 = X_train[[Y_train==1]].index
    pp_train_idx = balanced_idx1.union(Y_train[Y_train==0].sample(n=balanced_idx1.size, random_state=1234).index)
    
    X_train_balanced = X_train.loc[pp_train_idx, :]
    Y_train_balanced = Y_train.loc[pp_train_idx]
    A_train_balanced = A_train.loc[pp_train_idx]
    
    postprocess_est.fit(X_train_balanced, Y_train_balanced, sensitive_features=A_train_balanced)
    
    postprocess_preds = postprocess_est.predict(X_test, sensitive_features=A_test)
     
    return postprocess_preds

def show_comparison(model, X_test, y_test, A_test, protected_features, prostprocess_preds):
    """
    Returns Dashboard to show comparison of models based on the trade off of the disparity and accuracy
    
    """
    FairlearnDashboard(sensitive_features=A_test, sensitive_feature_names=protected_features,
                       y_true=Y_test,
                       y_pred={"Unmitigated": model.predict(X_test) ,
                              "ThresholdOptimizer": postprocess_preds})
                    
    return dashboard
