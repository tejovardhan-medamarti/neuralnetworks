import pandas as pd
import numpy as np


df=pd.read_csv('data/Admission_Predict_Ver1.1.csv')
# print(df.head())
df.info()
print(df.duplicated().sum())
df.drop(columns=['Serial No.'], inplace=True)

#  Lets use min max scaler to scale the data
X=df.iloc[:,:-1]
y=df.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20, random_state=1)
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
print(X_train_scaled)

#build Neural network architecture.

import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense

model=Sequential()

model.add(Dense(7,activation='relu',input_dim=7))
model.add(Dense(7,activation='linear')) # when working with regression problem, we use linear activation function
model.summary()


model.compile(optimizer='adam',loss='mean_squared_error',metrics=['mean_squared_error'])    # for regression problem, we use mean squared error as loss function
history = model.fit(X_train_scaled,y_train,epochs=10,validation_split=0.2)

y_pred=model.predict(X_test_scaled) # predict the test data
print(y_pred)
from sklearn.metrics import r2_score
# rs = r2_score(y_test,y_pred) # calculate r2 score
# print(rs)

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.savefig('output/loss_gradadm.png')
plt.close()
