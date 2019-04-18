import numpy as np
import time
from tqdm import tqdm

"""
Class used to implement logistic regression
"""
class LogisticRegression:

    def __init__(self,learning_rate=0.01,xShape=300):
        self.weights = np.zeros(xShape+1) # we add one for the intercept column
        self.learning_rate = learning_rate

    """
    Compute classification
    """
    def sigmoid(self,X):
        z = np.dot(X, self.weights)
        return 1 / (1 + np.exp(-z))

    """
    Compute cost of error for the logistic regression prediction
    """
    def cost(self, h, y):
        """
        y is always 0 or 1
        """
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    """
    Train the weights of the logistic regression
    """
    def train(self,X,Y,epochs = 1):

        X = np.array(X)
        Y = np.array(Y)

        start = time.time()

        # add the intercept column
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

        evolutionTrainingLoss = []

        for i in tqdm(list(range(epochs))):

            h = self.sigmoid(X)

            if i == epochs-1:
                print("Cost: {}".format(self.cost(h,Y)))
            if epochs % 100 ==0:
                evolutionTrainingLoss.append(self.cost(h,Y))

            # Now adjust weights to fit with data using gradient descent
            transposed = X.T
            gradient = np.dot(transposed, (h - Y)) / Y.size
            self.weights -= self.learning_rate * gradient

        print("Model trained in {} seconds with {} epoch(s).".format(time.time() - start,epochs))
        return evolutionTrainingLoss

    """
    Returns a Y matrix with predictions.
    As result is supposed ot be 1 or 0, round proba of 1
    """
    def predict(self,X):

        X = np.array(X)

        # add the intercept column
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

        result = self.sigmoid(X)

        return [1 if value > 0.5 else 0 for value in result]
