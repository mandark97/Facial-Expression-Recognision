import sys

from utils import load


class Trainer(object):

    def __init__(self,config):
        self.model = config.model()

    def train(self):
        X, y = load(mode=0)
        X_validation, y_validation = load(mode=1)
        self.model.train(X, y, X_validation, y_validation)

    def evaluate(self):
        X_test, y_test = load(mode=2)
        self.model.evaluate_model(X_test, y_test)
