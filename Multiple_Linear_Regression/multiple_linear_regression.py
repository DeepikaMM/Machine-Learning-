
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

#encoding catagorical data
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#avoiding dummy variable trap
X = X[:,1:]


#splitting the dataset into train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3,random_state = 0)

#feature scaling  using either standardization or normalization
#mostof thesimpe linear regression algo will take care of feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

#fitting linear regression to the traning data
from sklearn.linear_model import LinearRegression
Regressor = LinearRegression()
Regressor.fit(X_train,y_train)

y_pred = Regressor.predict(X_test)

#Adding bo = 1 i.e making xo = 1 , look atthe multiple liner regression equation
#building optimal model using backward elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X  , axis= 1 )

#backward elimination ALL-in
X_opt = X[:,[0,1,2,3,4,5]]

#significance level slected and fit the full model with allpossible independent variable
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() 

#getting the p value of each independent vlues
regressor_OLS.summary()

#fit the model  excaptinh high p value variable
X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() 
regressor_OLS.summary()

X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() 
regressor_OLS.summary()

X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() 
regressor_OLS.summary()

X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() 
regressor_OLS.summary()

#predicting using X_opt
# Splitting the dataset into the Training set and Test set
X_opt_train, X_opt_test, y_opt_train, y_opt_test = train_test_split(X_opt, y, test_size = 1/3, random_state = 0)
regressor_opt = LinearRegression()
regressor_opt.fit(X_opt_train, y_opt_train)
y_opt_pred = regressor_opt.predict(X_opt_test)




















