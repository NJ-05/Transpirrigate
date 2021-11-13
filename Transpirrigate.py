import sklearn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('tdataset.csv')
X = dataset.iloc[:, 0:3].values
y = dataset.iloc[:, 3].values
   
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

import statsmodels.api as sm

X = np.append(arr = np.ones((4404,1)).astype(int), values = X, axis = 1)

X_opt = X[:,[0,1,2,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
coeffs=regressor_OLS.params

amt_s = float(input("Enter Amount of Water Supplied (in ml): "))
temp = float(input("Enter Ambient Temperature (in Celsius):"))
temp += 273.15
hum = float(input("Enter Ambient Humidity (%):"))
hum /= 100

amt_t = np.dot(coeffs, [1, amt_s, temp, hum])

if amt_t < 0.9:
    amt_t = 0
    
print("The amount of water transpired is",round(amt_t,2),"ml")
if amt_t == 0:
    print("The amount of transpired water is negligible")
print("Tomorrow, supply",round((amt_s-amt_t),2),"ml to the plant")
