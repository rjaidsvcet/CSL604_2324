import numpy as np

class MultipleRegression:
    def __init__ (self):
        self.params = np.zeros(int(np.random.random()), float)[:,np.newaxis]

    def fit (self, X, y):
        bias = np.ones (len (X))
        X_bias = np.c_[bias, X]
        inner_part = np.transpose (X_bias) @ X_bias
        inverse_part = np.linalg.inv (inner_part)
        outer_part = inverse_part @ np.transpose (X_bias)
        lse = outer_part @ y
        self.params = lse
        return self.params
    
    def predict (self, X):
        bias_testing = np.ones (len (X))
        X_test = np.c_[bias_testing, X]
        y_hat = X_test @ self.params
        return y_hat
    
if __name__ == '__main__':

    X = np.array ([
        [1, 4],
        [2, 5],
        [3, 8],
        [4, 2]
    ])

    y = np.array ([1, 6, 8, 12])

    model = MultipleRegression ()
    parameters = model.fit (X, y)
    print (f'The parameters are {parameters}')
    y_pred = model.predict ([[5, 3]])
    print (f'The predicted value is {y_pred}')