import pandas as pd

import pickle

def load_trained_model(model_trained):

    # load model from pickle file
    with open(model_trained, 'rb') as file:  
        model = pickle.load(file)

    return model
 


def load_preprocessor():

    with open("preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)

    return preprocessor



def convert_house_df(house_data):
    df_house = pd.DataFrame(house_data)

    return df_house




def transform_data_house(df_house, preprocessor):
    transformed_house = preprocessor.transform(df_house)

    return transformed_house



def predict_price(transformed_house, model):

    prediction = model.predict(transformed_house)
    
    return prediction





model_trained = "notebooks/models/random_forest_model.pkl"

model = load_trained_model(model_trained)

preprocessor = load_preprocessor()



house_data = [{
    "Locality name": "Aalst",
    "Postal code": 9300,
    "Type of property": "House",
    "Subtype of property": "Detached",
    "Number of rooms": 3,
    "Living area": 125,
    "Equipped kitchen": "Installed",
    "Furnished": 0,
    "Open fire": 0,
    "Terrace": 1,
    "Garden": 1,
    "Number of facades": 4,
    "Swimming pool": 0,
    "State of building": "Good",
    "Garden Surface": 90,
    "Terrace Surface": 15
}]



df_house = convert_house_df(house_data)


transformed_house = transform_data_house(df_house, preprocessor)

prediction = predict_price(transformed_house, model)

print(prediction)