import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
print(X_train.shape, y_train.shape)
print(X_train[0].shape) # 28x28 image

import matplotlib.pyplot as plt
plt.imshow(X_train[2])#, cmap='gray') 
plt.show()
plt.savefig('output/mnist.png')
X_train = X_train / 255
X_test = X_test / 255
print(X_train[0])

# Lets creat ANN
model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_split=0.2)

y_prob = model.predict(X_test)
print(y_prob)
y_pred = y_prob.argmax(axis=1)

from sklearn.metrics import accuracy_score
acc_score =accuracy_score(y_test, y_pred)
print(acc_score)