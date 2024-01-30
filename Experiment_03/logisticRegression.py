import numpy as np

class LogisticRegression:
    def __init__ (self):
        self.params = np.zeros(int(np.random.random()), float)[:,np.newaxis]

    def fit (self, X, y):
        bias = np.ones (len (X))
        X_bias = np.c_[bias, X]
        inner_part = np.transpose (X_bias) @ X_bias
        inverse_part = np.linalg.inv (inner_part)
        outer_part = inverse_part @ np.transpose (X_bias)
        least_square_estimate = outer_part @ y
        self.params = least_square_estimate
        return self.params
    
    def predict (self, X):
        bias_testing = np.ones (len (X))
        X_test = np.c_[bias_testing, X]
        z = X_test @ self.params
        sigmoid = 1 / (1 + np.exp (-z))
        if sigmoid >= 0.5:
            y_hat = 1
        else:
            y_hat = 0
        return sigmoid, y_hat

if __name__ == '__main__':
    X = np.array([.50, 1.50, 2.00, 4.25, 3.25, 5.50], ndmin=2).reshape((6,1))
    y = np.array([0, 0, 0, 1, 1, 1])
    
    model = LogisticRegression ()
    parameters = model.fit (X, y)

    sig, y_pred = model.predict ([[3.75]])
    print (f'The predicted outcome is {y_pred} and calculated sigmoid value is {sig}')