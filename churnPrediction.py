import numpy as np
import pandas as pd 


# Read the data
df = pd.read_csv("data/Churn_Modelling.csv")
# print(df.head())
# print(df.shape)

df.info()

print(df.duplicated().sum())

print(df['Exited'].value_counts())
# DAta is imbalanced

print(df['Geography'].value_counts())
print(df['Gender'].value_counts())

df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)

# print(df.head())    
# convert the categorical data into numerical data, it is done by using one hot encoding

df =pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)
# print(df.head())

from sklearn.model_selection import train_test_split
X = df.drop(columns=['Exited'])
y = df['Exited']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# scaling the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train_scaled)





