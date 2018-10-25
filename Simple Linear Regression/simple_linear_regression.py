    
#getting the data and preprocessing it

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values



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

#predicting the test set result
y_pred= Regressor.predict(X_test)

#visualizing the traning set result
plt.scatter(X_train,y_train, color = 'red')
plt.plot(X_train,Regressor.predict(X_train), color= 'blue')
plt.title('salary vs Experiance(Training set)')
plt.xlabel('yesrs of experiance')
plt.ylabel('salary')
plt.show()

#visualizing test set result

plt.scatter(X_test,y_test, color = 'red')
plt.plot(X_train,Regressor.predict(X_train), color= 'blue')
plt.title('salary vs Experiance(Testing set)')
plt.xlabel('yesrs of experiance')
plt.ylabel('salary')
plt.show()









