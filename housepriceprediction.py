import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import sklearn.datasets
from xgboost import XGBRegressor
from sklearn import metrics



dataset=pd.read_csv(r'C:\Users\thenn\OneDrive\Desktop\HousingData.csv')
dataset.rename(columns={'MEDV': 'Price'}, inplace=True)

print(dataset.head(5))


dataset=dataset.dropna()

print(dataset.isnull().sum())

print(dataset.shape)

correlation=dataset.corr()
#plt.figure(figsize=(10,10))
#sns.heatmap(correlation,cbar=True,square=True,fmt='.1f',annot=True,annot_kws={'size':8},cmap='Blues')

#plt.show()

x=dataset.drop(columns='Price',axis=1)
y=dataset['Price']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
print(x_train.shape,x_test.shape,x.shape)

model=XGBRegressor()
model.fit(x_train,y_train)

training_data_prediction=model.predict(x_train)
#r square error
score1=metrics.r2_score(y_train,training_data_prediction)

#mean absolute error

score2=metrics.mean_absolute_error(y_train,training_data_prediction)

print(f'r square error:{score1}\n mean square error:{score2}')

prediction=model.predict(dataset.iloc[1,:-1].values.reshape(1,-1))
print(f'prediction:  {prediction[0]}')