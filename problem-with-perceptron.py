from operator import xor
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns

or_data =pd.DataFrame()
and_data =pd.DataFrame()
xor_data =pd.DataFrame()

or_data['input1'] = [1,1,0,0]
or_data['input2'] = [1,0,1,0]
or_data['output'] = [1,1,1,0]

and_data['input1']=[1,1,0,0]
and_data['input2']=[1,0,1,0]
and_data['output']=[1,0,0,0]

xor_data['input1']=[1,1,0,0]
xor_data['input2']=[1,0,1,0]
xor_data['output']=[0,1,1,0]

print(and_data)

from sklearn.linear_model import Perceptron

clf1=Perceptron()
clf2=Perceptron()
clf3=Perceptron()

clf1.fit(and_data.iloc[:,0:2].values,and_data.iloc[:,1].values)
clf2.fit(or_data.iloc[:,0:2].values,or_data.iloc[:,1].values)   
clf3.fit(xor_data.iloc[:,0:2].values,xor_data.iloc[:,1].values)

print(clf1.coef_, clf1.intercept_)
print(clf2.coef_, clf2.intercept_)
print(clf3.coef_, clf3.intercept_)

x=np.linspace(-1,1,5)
y=-x+1

plt.plot(x,y)

sns.scatterplot(data=and_data, x="input1", y="input2", hue="output")
plt.savefig('output/and.png')

plt.close()

x1=np.linspace(-1,1,5)
y1=-x1+0.5

plt.plot(x1,y1)
sns.scatterplot(data=or_data, x="input1", y="input2", hue="output") 
plt.savefig('output/or.png')
plt.close()

from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X=xor_data.iloc[:,0:2].values, y=xor_data.iloc[:,2].values, clf=clf3, legend=2)   
plt.savefig('output/xor.png')
plt.close()
