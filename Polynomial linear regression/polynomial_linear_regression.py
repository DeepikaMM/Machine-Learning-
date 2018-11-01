#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 14:30:53 2018
Bluffing Detector :-P
@author: deepika
"""

#polnomial regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer

#importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#splitting data  into train and test data set no need coz it small
'''from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state = 0)'''

#fitting linear regression to thedata set
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)


#fitting polynomial regression to thedata set
#change degree make it fit to the mode for better accuracy
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

lin_reg2  = LinearRegression()
lin_reg2.fit(X_poly,y)

#visualizing the set result
plt.scatter(X,y, color = 'red')
plt.plot(X,lin_reg.predict(X), color= 'blue')
plt.title('Truth of Bluff(Linear regression)')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()

#visualizing the polynomil set result
#creating continous curve
#added later
#before this ry replacing X_grid with X_poly
X_grid = np.arange(min(X),max(X),0.1)
X_grid =X_grid.reshape(len(X_grid),1)
plt.scatter(X,y, color = 'red')
plt.plot(X_grid,lin_reg2.predict( poly_reg.fit_transform(X_grid)), color= 'blue')
plt.title('Truth of Bluff(polynomial regression)')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()


#predicting a new result with linear regresssion
lin_reg.predict(6.5)
#predicting a new result with polynomial regresssion
lin_reg2.predict( poly_reg.fit_transform(6.5))




