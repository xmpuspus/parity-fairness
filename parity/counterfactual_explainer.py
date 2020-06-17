# pip install dice-ml
import pandas as pd
import dice_ml

#from dice_ml.utils import helpers # helper functions

def get_data_object(data, continuous_features, to_predict):
    
    """
    Gets the required features about the data to be used for getting the countefactuals.
    
    Parameters:
    data: the whole dataset which includes the training and testing dataset
    continuous_features: list of names of features that assumes all the possible values in a continuum
    to_predict: the name of the variable to be predicted.
    
    Returns:
    data_object (object): parameters about the data such as such as the range of continuous features and the levels of categorical features.   
    
    """    
    data_object = dice_ml.Data(dataframe=data, continuous_features=continuous_features, outcome_name=to_predict)
    
    return data_object

def get_explainer_object(model_path, model_backend, data_object):
    """
    Provides feature importances to explain the model.
    
    Parameters:
    model: trained model
    model_backend: indicates the implementation type of DiCE we want to use.
    data_object: DiCE data object
    
    Returns:
    explainer (object): provides the feature importances that determines the prediction of the model
    
    """
    model_object = dice_ml.Model(model_path=model_path, backend=model_backend) 
    
    explainer = dice_ml.Dice(data_object, model_object)
    
    return explainer
    
def generate_counterfactual(ready_object, query_instance, number_CF, desired_pred, 
                             feature_weights, proximity_weight, diversity_weight, feature_to_vary):

    """
    Generate counterfactual profiles with feature-perturbed versions.
    
    Parameters:
    ready_object: the DiCE class
    query_instance: a query input whose outcome needs to be explained. 
                    query instance shoulde be in the form of a dictionary; keys: feature name, values: feature value
    number_CF: total number of counterfactuals to generate
    desired_pred: the desired outcome of prediction
    feature_weights: a dictionary; keys: continuous feature name, values: weights. 
    proximity_weight: weight for the counterfactuals be closer and feasible to an individual's profile(query instance)
    diversity_weight: weight for the counterfactuals be providing the individual multiple options
    feature_to_vary: a list of features that are allowed to vary since other suggested features are not easily be varied

    Returns:
    
    viz(dataframe): profiles with feature-perturbed versions that will produce a desired prediction
    
    """
    exp = ready_object
    dice_exp = exp.generate_counterfactuals(query_instance = query_instance, total_CFs=number_CF,
                                            desired_class=desired_pred, diversity_weight = diversity_weight)
                                           
    # Visualize counterfactual explanation
    viz = dice_exp.visualize_as_dataframe()
    
    return viz
