import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split 
from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

import pickle




# ---------------------------------------------------
# 1) CLEAN DATA
# ---------------------------------------------------

def clean_data(datasheet):

    # We make a copy to work more securely

    df_clean = datasheet.copy()

    # We removed the column 'Property ID'
    df_clean = df_clean.drop(columns=['Property ID'])



    column_NaN = [] # a list of the columns that will be removed


    for column in df_clean.columns:
        amount_NaN = df_clean[column].isnull().sum()

        percentage = amount_NaN * 100 / df_clean.shape[0]

        if percentage > 70:
            
            column_NaN.append(column) # add the column name to the list



    # We removed the identified columns
    for col in column_NaN:
        df_clean = df_clean.drop(columns=[col])


    return df_clean




# ---------------------------------------------------
# 2) PREPROCESSING PIPELINE
# ---------------------------------------------------


def preprocessing(df_clean):


    y = df_clean['Price'] # we define 'y' value, the 'target'

    # X values
    X = df_clean.drop(columns=['Price'])

    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size=0.2)



    # ------------------------------------
    # IMPUTATION
    # ------------------------------------

    dict_imputers ={}

    for X_column in X_train.columns:

        data_type = X_train[X_column].dtype

        if np.issubdtype(data_type, np.number):

            imputer = SimpleImputer(missing_values=np.nan, strategy='median')

            imputer.fit(X_train[[X_column]])   # calculate the median
            dict_imputers[X_column] = imputer

            imputer_values = imputer.transform(X_train[[X_column]])  # NaN are replaced by the calculated value
            X_train[X_column] = imputer_values     # reassign that entire column



        if data_type == object:
            
            imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

            imputer.fit(X_train[[X_column]])   # calculate the mode
            dict_imputers[X_column] = imputer

            imputer_values = imputer.transform(X_train[[X_column]])  # NaN are replaced by the calculated value
            
            X_train[X_column] = imputer_values.ravel() # reassign that entire column and change to 1D


    for X_column in X_test.columns:

        if X_column in dict_imputers:

            imputer = dict_imputers[X_column]
            X_test[X_column] = imputer.transform(X_test[[X_column]]).ravel() # reassign that entire column and change to 1D




    # ------------------------------------
    # ADJUST THE COLUMNS
    # ------------------------------------

    # We created two lists to identify which columns are categorical or numeric.

    categorical_cols = []
    numeric_cols = []


    for X_column in X_train.columns:

        data_type = X_train[X_column].dtype

        if data_type in ['float64', 'int64']:
            numeric_cols.append(X_column)

        elif data_type == object:
            categorical_cols.append(X_column)


    ohe = OneHotEncoder(sparse=True, handle_unknown='ignore')


    ct = ColumnTransformer([
        ('categorical', ohe, categorical_cols),
        ('numerical', StandardScaler(), numeric_cols)
    ])


    ct.fit(X_train) # learn parameters
    X_train = ct.transform(X_train) # Applies the transformations and returns the processed data

    X_test = ct.transform(X_test) # Use the same parameters learned from train

    return X_train, X_test, y_train, y_test





def save_preprocessed_data(X_train, X_test, y_train, y_test):

    # Save the preprocessed data
    with open("preprocessed_data.pkl", "wb") as f:
        pickle.dump((X_train, X_test, y_train, y_test), f)



'''

datasheet = pd.read_csv('data/data.csv')

cleaned_df = clean_data(datasheet)


X_train, X_test, y_train, y_test = preprocessing(cleaned_df)

save_preprocessed_data(X_train, X_test, y_train, y_test)

'''
