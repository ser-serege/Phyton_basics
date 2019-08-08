x = int(input())
y = int(input())
z = int(input())

p = (x + y + z) / 2
s = (p*(p-x)*(p-y)*(p-z))**(1/2)

print('{0:.6f}'.format(s))


from sklearn import datasets
import numpy as np
import pandas as pd

iris = datasets.load_iris()
X = iris.data[:, :2]
y = (iris.target != 0) * 1

def Logistic_regression(X, y,  num_iter = 100,  learning_rate = 1, X_test = None, probs = True):

    def sigmoid(X, weight): # final activation function 
        z = np.dot(X, weight) # product of matrix X and weights 
        return 1 / (1 + np.exp(-z)) # sigmoid function to predict probability  

    def log_likelihood(X, y, weights): #
        z = np.dot(X, weights) # product of matrix X and weights 
        ll = np.sum( y * z - np.log(1 + np.exp(z)) )
        return ll

    def gradient_ascent(X, sigmoida, y): # 
        return np.dot(X.T, y - sigmoida)

    def update_weight_mle(weight, learning_rate, gradient):
        return weight + learning_rate * gradient

    def fit(X, y, num_iter ,  learning_rate ):
        intercept = np.ones((X.shape[0], 1)) # initialize intercept  
        X = np.concatenate((intercept, X), axis=1) # put intercept into dataset
        theta = np.zeros(X.shape[1]) # initialize weights
        for i in range(num_iter): # define num of iterations
            sigmoida = sigmoid(X, theta)  # 1 / (1 + np.exp(- (np.dot(X, weight)))) - calc answers
            gradient = gradient_ascent(X, sigmoida, y) #np.dot(X.T, (h - y)) / y.size - calc gradient
            theta = update_weight_mle(theta, learning_rate, gradient) # weight + learning_rate * gradient - make step
        return theta, intercept

    def predict_proba(X, theta): # define probability
        intercept = np.ones((X.shape[0], 1))
        X = np.concatenate((intercept, X), axis=1) # concat with intercept
        return sigmoid(X, theta)

    def predict(X, theta, threshold=0.5): # define 1 or 0 with threshold
        return predict_proba(X, theta) >= threshold 
    
    theta, intercept = fit(X, y, num_iter, learning_rate)

    if probs:    
        result = predict_proba(X, theta)

        try:
            result2 = predict_proba(X_test, theta)
            print('Train predicted')
            print('Test predicted')
            return result, result2
        except: result2 = None

        print('Train predicted')
        print('Test not predicted')
    
    
    else:
        result = predict(X, theta)

        try:
            result2 = predict(X_test, theta)
            print('Train predicted')
            print('Test predicted')
            return result, result2
        except: result2 = None

        print('Train predicted')
        print('Test not predicted')

    return result, result2

def train_test_split(X, train_size = 0.7):
    ind = int(len(X) * train_size)

    X_train = X[:ind]
    y_train = y[:ind]

    X_test = X[ind:]
    y_test = y[ind:]

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split(X, train_size = 0.7)

result_train, result_test = Logistic_regression(X_train, y_train , X_test= X_test, probs= False)
print('train=', (result_train == y_train).mean())
print('test=', (result_test == y_test).mean())

result_train, result_test = Logistic_regression(X_train, y_train , X_test= False, probs= False)
print('train=', (result_train == y_train).mean())
print('test=', (result_test == y_test).mean())

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)

result_sklearn_train = lr.predict(X_train)
result_sklearn_test = lr.predict(X_test)

print('train=', (result_sklearn_train == y_train).mean())
print('test=',  (result_sklearn_test == y_test).mean())
	
	


	
