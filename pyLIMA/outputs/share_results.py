import pickle

def save_results(model_object, fit_object, filename, run_silent=True):
    """
    Serializes and saves a model object and a fit object to a file.
    
    Parameters:
    - model_object: The pyLIMA model object to be saved.
    - fit_object: The pyLIMA fit object related to the model object to be saved.
    - filename: String specifying the path to save the pickle file.
    
    Returns:
    - None
    """
    with open(filename, 'wb') as outfile:
        pickle.dump((model_object, fit_object), outfile)
    if not run_silent:
        print ('Saved', model_object.model_type,'model fit results for event', 
               model_object.event.name)

def load_results(filename, run_silent=True):
    """
    Deserializes model and fit objects from a file and prints model information.
    
    Parameters:
    - filename: String specifying the path to the pickle file containing the 
      pyLIMA model and fit serialized objects.
    
    Returns:
    - model_object: The deserialized model object.
    - fit_object: The deserialized fit object associated with the model object.
    
    Prints the model type and the event name of the model object.
    """
    with open(filename, 'rb') as infile:
        model_object, fit_object = pickle.load(infile)
    if not run_silent:
        print (model_object.model_type,'model fit results loaded for event', 
               model_object.event.name)
    return model_object, fit_object
