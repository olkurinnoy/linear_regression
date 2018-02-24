import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #for 3D plots
from math import ceil
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

#load data
location = '../data/data.csv'
data = np.genfromtxt(location, dtype = int, delimiter=',')
X = data[:,[0,1]]
Y = data[:, 2]

#no need to use normalization 

#merge input data into training and test set in proportion 4:1 
#(better include one more cross-validated set but no needing in this ML model)
m_train = int(float(X.shape[0]*4)/5)
m_test = X.shape[0] - m_train

X_train = X[:m_train,:]
Y_train = Y[:m_train]

X_test = X[m_train + 1:,:]
Y_test = Y[m_train + 1:]

#creating linear regression model
regr = linear_model.LinearRegression()

#train model on training set (obtaining vector theta)
regr.fit(X_train, Y_train)

#predicting output values for test set
Y_test_pred = regr.predict(X_test)

#computing prediction error as cost function
pred_error = mean_squared_error(Y_test_pred, Y_test)

#computing learning error
Y_train_pred = regr.predict(X_train)
learn_error = mean_squared_error(Y_train_pred, Y_train)

print pred_error, learn_error

#drawing plots -- learning curve on training set and test set
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train[:,0], X_train[:,1], Y_train, c='r', marker='o')

ax.plot(X_train[:,0], X_train[:,1], Y_train_pred, c='b')

ax.set_xlabel('Positive sent')
ax.set_ylabel('Negative sent')
ax.set_zlabel('Score')
plt.title('Training accuracy')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test[:,0], X_test[:,1], Y_test, c='r', marker='o')

ax.plot(X_test[:,0], X_test[:,1], Y_test_pred, c='b')

ax.set_xlabel('Positive sent')
ax.set_ylabel('Negative sent')
ax.set_zlabel('Score')
plt.title('Test accuracy')
plt.show()