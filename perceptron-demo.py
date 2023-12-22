import matplotlib
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/placementsample.csv')
print(df.shape)
df.head()

sns.scatterplot(data=df, x="cgpa", y="resume_score", hue="placed")
plt.show()
x=df.iloc[:,0:2]
y=df.iloc[:,-1]
print(x)
print(y)


from sklearn.linear_model import Perceptron
p=Perceptron()
p.fit(x,y)
print(p.coef_)
print(p.intercept_)

from mlxtend.plotting import plot_decision_regions
plot_decision_regions(x.values, y.values, clf=p,legend=2)

plt.show()