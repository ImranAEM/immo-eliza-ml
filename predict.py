import pickle

def load_trained_model(model_trained):

    # load model from pickle file
    with open(model_trained, 'rb') as file:  
        model = pickle.load(file)

    return model
 





model_trained = "RandomForest_train_model.pkl"
model = load_trained_model(model_trained)
