import pickle

from sklearn.ensemble import RandomForestRegressor



def load_data(preprocessing_data):

    # load model from pickle file
    with open(preprocessing_data, 'rb') as file:  
        data = pickle.load(file)

    return data
 


def model_train(model, X_train, y_train):

   model_trained = model.fit(X_train, y_train)

   return model_trained



def save_model(trained_model, filename):

    with open(filename, "wb") as file:
        pickle.dump(trained_model, file)






# Upload the preprocessed data

data = load_data("preprocessed_data.pkl")
X_train, X_test, y_train, y_test = data


# Train the model

random_for = RandomForestRegressor()

trained_model = model_train(random_for,X_train, y_train)


# Save the trained model

filename = "RandomForest_train_model.pkl"
save_model(trained_model, filename)

