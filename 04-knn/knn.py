class KNeighborsClassifier:

    def __init__(self, k):
        self.k = k

    def fit(self, X_train, y_train):
        raise NotImplementedError()

    def predict(self, X_test):
        raise NotImplementedError()

    def score(self, X_test, y_test):
        raise NotImplementedError()
