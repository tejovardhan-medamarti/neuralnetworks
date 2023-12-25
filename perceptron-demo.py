import pandas as pd
import seaborn as sns
# import matplotlib
# matplotlib.use('TkAgg') 
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
plt.savefig('output/perceptrons-1.png')
# 
# 
from sklearn.datasets import make_classification
import numpy as np
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=1,
                           n_classes=2,random_state=41, n_clusters_per_class=1, 
                           hypercube=False, class_sep=10)
plt.figure(figsize=(10,6))
plt.scatter(X[:,0],X[:,1],c=y, cmap='winter', s=100)

plt.savefig('output/makeClassification.png')

print(X.shape)

def step(z):
    return 1.0 if z>=0.0 else 0.0

def Perceptron(X,y):
    X=np.insert(X,0,1,axis=1)
    weights=np.ones(X.shape[1])
    lr = 0.1
    for i in range(1000):
        j= np.random.randint(0,X.shape[0])
        y_hat = step(np.dot(X[j],weights))
        weights += lr*(y[j]-y_hat)*X[j]
    return weights[0],weights[1:]

intercept_, coeff_ = Perceptron(X,y)

print("Wo:", intercept_, "W1 and W2",coeff_)

m= -(coeff_[0]/coeff_[1])
b= -(intercept_/coeff_[1])

x_input = np.linspace(-3,3,100)
y_input = m*x_input + b

plt.figure(figsize=(10,6))
plt.plot(x_input,y_input,color ='red', linewidth =3)
plt.scatter(X[:,0],X[:,1],c=y, cmap='winter', s=100)
plt.ylim(-3,2)
plt.savefig('output/makeClassificationwithln.png')
plt.show()




def Perceptron_Animation(X,y):
    m= []
    b=[]

    X=np.insert(X,0,1,axis=1)
    weights=np.ones(X.shape[1])
    lr = 0.1
    for i in range(200):
        j= np.random.randint(0,X.shape[0])
        y_hat = step(np.dot(X[j],weights))
        weights += lr*(y[j]-y_hat)*X[j]
        m.append(-(weights[1]/weights[2]))
        b.append(-(weights[0]/weights[2]))

    return m,b  

intercept_, coeff_ = Perceptron(X,y)
m,b = Perceptron_Animation(X,y)

from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation


fig, ax = plt.subplots(figsize=(10,6))

x_i=np.arange(-3,3,0.1)
y_i=x_i*m[0]+b[0]
ax.scatter(X[:,0],X[:,1],c=y, cmap='winter', s=100)
line, = ax.plot(x_i,y_i,color ='red', linewidth =3)
plt.ylim(-3,3)

def update(i):
    label='epoch{0}'.format(i+1)
    line.set_ydata(x_i*m[i]+b[i])
    ax.set_xlabel(label)

def perceptron_gradient(X,y):
    w1=w2=b=1
    lr = 0.1
    for j in range(1000): #epochs
        for i in range(X.shape[0]):

            z=w1*X[i][0]+w2*X[i][1]+b

            if z*y[i]<0:    #misclassified/ update weights
                w1 += lr*y[i]*X[i][0]
                w2 += lr*y[i]*X[i][1]
                b += lr*y[i]
        
    return w1,w2,b


ani = FuncAnimation(fig, update,repeat=True, frames=np.arange(0, len(m)), interval=100) 
ani.save('output/makeClassificationanimation.gif', writer='imagemagick', fps=10)


def calculate_Slope(w1,w2,b):
    m= -(w1/w2)
    c= -(b/w2)
    return m,c

w1,w2,b = perceptron_gradient(X,y)
print(w1,w2,b)
m,c = calculate_Slope(w1,w2,b)

x_input = np.linspace(-3,3,100)
y_input = m*x_input + c

plt.figure(figsize=(10,6))
plt.plot(x_input,y_input,color ='red', linewidth =3)
plt.scatter(X[:,0],X[:,1],c=y, cmap='winter', s=100)
plt.ylim(-3,2)
plt.savefig('output/makeClassificationwithgradient.png')
