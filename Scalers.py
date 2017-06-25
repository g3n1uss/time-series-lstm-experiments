import numpy as np

class NewScaler:
    def __init__(self, params):
        self.params = params



class TanhScaler:
    def __init__(self, max_val_at=None, parameter=None):
        """
        We want tanh(parameter*x_max) = max_val_at
        So we can either specify 'parameter' directly or compute it inside 'fit'

        :param parameter:
        :param max_val_at:
        """
        self.max_val_at = max_val_at
        self.parameter = parameter

    def fit(self, X):
        """
        Solve tanh(parameter*x_max) = max_val_at for 'parameter'

        :param X:
        :return:
        """
        if self.max_val_at is None:
            print("Specify 'max_val_at' or 'parameter'")
            return
        X_max = np.max(X)
        self.parameter = np.arctanh(self.max_val_at)/X_max
        return True


    def transform(self, X):
        if self.max_val_at is not None:
            self.fit(X)
        return np.tanh(self.parameter * X)

    def inverse_transform(self, X):
        if self.parameter is None:
            print('Fit first')
            return
        else:
            return np.arctanh(X)/self.parameter

'''
# test
X = np.array([0, 1, 343.34, 600])
# scaler = TanhScaler(None, 0.0024537)
scaler = TanhScaler(0.9)
normalized = scaler.transform(X)
print(normalized)
print(scaler.inverse_transform(normalized))
'''