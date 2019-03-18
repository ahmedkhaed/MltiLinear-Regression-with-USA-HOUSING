# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 23:39:00 2019

@author: Ahmed Khaled
steps for multi_linear regression are :
    step 1 :import libararies
    step 2 :Get data set
    step 3 :split data into input & output
    step 4 :check missing data
    step 5: check categeorical data
    step 6 :split data into training data  & test data 
    step 7 :Build your model
    step 8 :plot best line 
    step 9 :Estimate Error 
"""

 # step 1 :import libararies
import numpy as np   # to make  mathmatical operation on metrices
import pandas as pd  #to read data
import matplotlib.pyplot as plt   #to show some graghs
import seaborn as sns    #for plot data
from sklearn.cross_validation import  train_test_split #to split data to train & test
from sklearn.linear_model import LinearRegression    #to import linear model 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder   #for categeorical data
from sklearn.metrics import mean_squared_error #to calculate MSE ,MAE ,RMSE

# step 2 :Get data set
path = 'C:\\Users\\Ahmed Khaled\Downloads\\my work (regression)\\5)MltiLinear-Regression-with-USA-HOUSING-DATA-master\\USA_Housing.csv'
data = pd.read_csv(path)
print('data : \n ',data)
print('data.head : \n ',data.head())
print('data.shape : \n',data.shape)
print('names of columns :\n',data.columns)
print('data.imnformation: \n ' ,data.info())
print('data.describe: \n ' ,data.describe())
sns.pairplot(data)
sns.distplot(data['Price']) #distrebution
data.corr()    #corrolations
sns.heatmap(data.corr()) 
sns.heatmap(data.corr(),annot=True) #relation with number

#step 3 :split data into input & output

x = data[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = data['Price']

#step 4 :check missing data 
# there is no missing data 

#step 5: check categeorical data
# there is no  categeorical data

#step 6 :split data into training data  & test data 
#from sklearn.model_selection import train_test_split
x_train, x_test ,y_train,y_test =train_test_split(x,y,test_size = 0.4  ,random_state = 101 )

#step 7 :Build your model
#from sklearn.linear_model import LinearRegression 
model = LinearRegression()
model.fit(x_train,y_train)
# print the intercept
print(model.intercept_) 
coeff_data = pd.DataFrame(model.coef_,x.columns,columns=['Coefficient'])  #make table between coef & x (input data) 
print('coeff_data : \n',coeff_data) 

#step 8 :plot best line accordding to your prediction
from sklearn.datasets import load_boston
boston  = load_boston()
print('boston.Keys\n',boston.keys()) 
print(boston['target'])  # target from keys of dictionary of boston
print(boston['feature_names'])  # feature_names from keys of dictionary of boston

y_pred = model.predict(x_test)
plt.scatter(y_test,y_pred)
sns.distplot((y_test-y_pred),bins=50) 
# Predict the Score (% Accuracy)

print('Train Score :', model.score(x_train,y_train))
print('Test Score:', model.score(x_test,y_test))


#step 9 :Estimate Error 
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test,y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))