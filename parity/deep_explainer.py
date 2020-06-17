# import libraries from interpret-community
from interpret.ext.blackbox import TabularExplainer
from interpret.ext.blackbox import MimicExplainer
from interpret_community.widget import ExplanationDashboard


# for mimic_explainer: one of the following four interpretable models used as a global surrogate to the black box model
from interpret.ext.glassbox import LGBMExplainableModel
from interpret.ext.glassbox import LinearExplainableModel
from interpret.ext.glassbox import SGDExplainableModel
from interpret.ext.glassbox import DecisionTreeExplainableModel

def shap_values(model, x_train, x_text, features, initialization_examples):
   
    """    
    Provides feature importances to explain the model.
    
    Parameters:
    x_train: input dataset to train the model
    x_test: test dataset
    model: trained model
    features: list of feature names. Optional, used if doing classification
    classes: list of output class labels or names. Optional, used if doing classification
    
    Returns:
    explainer (object): provides the feature importances that determines the prediction of the model
    global_explanation (object): provides the global feature importances that determines the prediction of the model
    local_explanation (object): provides the global feature importances that determines the prediction of the model
    
    """
    explainer = TabularExplainer(model, x_train, features=features)     
    
    # you can use the training data or the test data here
    global_explanation = explainer.explain_global(x_test)
    
    # explain the selected data point in the test set
    local_explanation = explainer.explain_local(x_test)
    
    return explainer, global_explanation, local_explanation

def mimic_values(x_train, x_test, model, features, augment_data, max_num_of_augmentations, explainable_model = LinearExplainableModel):
    """  
    Parameters:
    
    Provides feature importances to explain the model using a surrogate model
    
    x_train: input dataset to train the model
    x_test: test dataset
    model: trained model
    explainable_model: interpretable model as a global surrogate to the black box model
    features: list of feature names. Optional, used if doing classification
    classes: list of output class labels or names. Optional, used if doing classification
    augment_data:is optional and if true, oversamples the initialization examples to improve surrogate model accuracy to fit originalmodel.                  Useful for high-dimensional data where the number of rows is less than the number of columns.
    max_num_of_augmentations: is optional and defines max number of times we can increase the input data size.
    
    Returns: 
    explainer (object): provides the feature importances that determines the prediction of the model
    global_explanation (object): provides the global feature importances that determines the prediction of the model
    local_explanation (object): provides the local feature importances that determines the prediction of the model
    
    """
    explainer = MimicExplainer(model, 
                               x_train, 
                               explainable_model, augment_data= augment_data, 
                               max_num_of_augmentations = max_num_of_augmentations, 
                               features=features)
    
    # you can use the training data or the test data here
    global_explanation = explainer.explain_global(x_test)
    
    # explain the selected data point in the test set
    local_explanation = explainer.explain_local(x_test)

    return explainer, global_explanation, local_explanation


def dashboard(x_test, model, explanation):
    """
    Returns viz (interactive dashboard) in your noteboook to understand the different features that explains the prediction of the model.
    
    Parameters:
    x_test: testing dataset
    model: trained model
    explanation: type of explaination, either local or global feature importances
    
    """
    viz = ExplanationDashboard(explanation, model, datasetX=x_test)
    
    return viz
