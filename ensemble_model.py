import numpy as np
from sklearn.svm import SVC


class EnsembleModel(object):
    def __init__(self, model_configs, load_weights=None):
        self.models = [config.model() for config in model_configs]

        # TODO complete with code
        if load_weights:
            pass

    def train(self, X_train, y_train):
        train_features = np.concatenate([model.extract_features(
            model.ensemble, X_train, y_train) for model in self.models])

        self.model = SVC(decision_function_shape='ovo', verbose=1)
        self.model.fit(train_features, y_train)

    def evaluate_model(self, X_test, y_test):
        test_features = np.concatenate([model.extract_features(
            model.ensemble, X_test, y_test) for model in self.models])
        self.model.score(test_features, y_test)

    def predict(self, x):
        features = np.concatenate([model.extract_features(
            model.ensemble, x, np.zeros(x.shape[0])) for model in self.models])
        # TODO complete with corect code
        return self.model.predict(features)
