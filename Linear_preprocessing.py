import pickle
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error, mean_squared_error


# Upload the data
with open("preprocessed_data.pkl", "rb") as f:
    X_train, X_test, y_train, y_test = pickle.load(f)


# Load and fit model

model_linear_regressor = LinearRegression()


# Train the model

model_linear_regressor.fit(X_train, y_train)

# Display the score of the model

model_linear_regressor.score(X_train, y_train)


# Test model

y_predict_test = model_linear_regressor.predict(X_test)


# Display the score

model_linear_regressor.score(X_test, y_test)



# MAE and MSE calculation


MAE_test = mean_absolute_error(y_test, y_predict_test)
MSE_test = mean_squared_error(y_test, y_predict_test)

print(f"MAE_test: {MAE_test}")
print(f"MSE_test:  {MSE_test}")