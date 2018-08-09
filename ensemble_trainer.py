from utils import load


class EnsembleTrainer(object):
    def __init__(self, ensemble_model):
        self.model = ensemble_model

    def train(self):
        X_train, y_train = load(mode=0)

        self.model.train(X_train, y_train)

    def evaluate(self):
        X_test, y_test = load(mode=2)
        self.model.evaluate_model(X_test, y_test)
