# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load California housing data, select features and targets, and split into training and testing sets. 2.Scale both X (features) and Y (targets) using StandardScaler. 3.Use SGDRegressor wrapped in MultiOutputRegressor to train on the scaled training data. 4.Predict on test data, inverse transform the results, and calculate the mean squared error.
## Program:
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: lavanya D
RegisterNumber:  212225040195
*/
# Program to implement SGD Regressor for Multivariate Linear Regression
# Developed by: SANTHIYA G
# Register Number:N21225230248

import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

X = np.array([
    [1000, 2],
    [1500, 3],
    [1800, 3],
    [2400, 4],
    [3000, 4]
])

Y = np.array([
    [50, 2],
    [75, 3],
    [90, 3],
    [120, 4],
    [150, 5]
])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = SGDRegressor(max_iter=1000, learning_rate='invscaling')

model.fit(X_scaled, Y[:, 0])
price_pred = model.predict(X_scaled)

model.fit(X_scaled, Y[:, 1])
occupant_pred = model.predict(X_scaled)

print("Predicted House Prices:", price_pred)
print("Predicted Number of Occupants:", occupant_pred)
```

## Output:
<img width="940" height="63" alt="image" src="https://github.com/user-attachments/assets/b7ddec44-e413-4157-85d4-4488512307ae" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
