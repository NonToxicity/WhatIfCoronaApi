import pandas as pd
import datetime
import pprint
import sklearn
from numpy import array
from numpy import argmax
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from collections import defaultdict
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
import pickle

df = pd.read_csv("owid-covid-data.csv")
df = df.fillna(0) 
df = df.applymap(str)
df.columns = df.columns.str.replace(' ', '')


from sklearn.utils import shuffle
df = shuffle(df)

# instantiate labelencoder object
le = LabelEncoder()

categorical_cols = ['location'] 

# apply le on categorical feature columns
df[categorical_cols] = df[categorical_cols].apply(lambda col: le.fit_transform(col))    
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()

#One-hot-encode the categorical columns.
#Unfortunately outputs an array instead of dataframe.
array_hot_encoded = ohe.fit_transform(df[categorical_cols])

#Convert it to df
data_hot_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

#Extract only the columns that didnt need to be encoded
data_other_cols = df.drop(columns=categorical_cols)

#Concatenate the two dataframes : 
data_out = pd.concat([data_hot_encoded, data_other_cols], axis=1)

data_out.to_csv('out.csv')  
data_out = data_out.applymap(float)
#data_out = data_out.applymap(int)
from sklearn.neighbors import KNeighborsClassifier

X = data_out.drop(columns=['total_cases'])
y = data_out['total_cases']

X.to_csv('x.csv')  

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.neighbors import KNeighborsRegressor
knn_regressor = KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
knn_regressor.fit(X_train,y_train)



kneigh_pred = knn_regressor.predict(X_test)
print('KNNREG -> Mean Absolute Error: ', metrics.mean_absolute_error(y_test, kneigh_pred))  
print('KNNREG -> R2 Score: ', r2_score(y_test, kneigh_pred))
print('KNNREG -> mean_absolute_percentage_error: ', mean_absolute_percentage_error(y_test, kneigh_pred))
print('KNNREG -> mean_squared_error: ', mean_squared_error(y_test, kneigh_pred))

# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(knn_regressor, open(filename, 'wb'))


#---------------------------------------------------------------------------------------------

from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_train, y_train)

reg_pred = reg.predict(X_test)
print('LR -> Mean Absolute Error: ', metrics.mean_absolute_error(y_test, reg_pred))  
print('LR -> R2 Score: ', r2_score(y_test, reg_pred))
print('LR -> mean_absolute_percentage_error: ', mean_absolute_percentage_error(y_test, reg_pred))
print('LR -> mean_squared_error: ', mean_squared_error(y_test, reg_pred))

print(X_test.iloc[0])

#score_neigh = cross_val_score(kneigh_pred, X, y, cv=5)
#print("Accuracy: %0.2f (+/- %0.2f)" % (score_neigh.mean(), score_neigh.std() * 2))