import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib
import os
import csv


class EnsembleModel(object):
    def __init__(self, ensemble_config):
        self.models = [config.model() for config in ensemble_config["models"]]
        self.model_name = ensemble_config["model_name"]

    def train(self, X_train, y_train):
        train_features = np.concatenate([model.extract_features(
            model.ensemble, X_train, y_train) for model in self.models])

        self.model = SVC(decision_function_shape='ovo', verbose=1)
        self.model.fit(train_features, y_train)

    def evaluate_model(self, X_test, y_test):
        test_features = np.concatenate([model.extract_features(
            model.ensemble, X_test, y_test) for model in self.models])
        score = self.model.score(test_features, y_test)

        if not os.path.exists(f'ensembles/self.model_name'):
            os.makedirs(f'ensembles/{self.model_name}')
        joblib.dump(self.model, f"ensembles/{self.model_name}/model.pkl")
        with open(f"ensembles/{self.model_name}/file.txt", 'w') as file:
            file.write(str(score))
        self._save_to_csv(score)

    def predict(self, x):
        features = np.concatenate([model.extract_features(
            model.ensemble, x, np.zeros(x.shape[0])) for model in self.models])

        return self.model.decision_function(features)

    def all_predictions(self, x):
        predictions = { model.model_name: model.predict(x) for model in self.models }

        predictions['ensemble'] = self.predict(x)

        return predictions

    def _save_to_csv(self, score):
        csv_name = 'results.csv'
        build_file = False

        with open(csv_name, 'r') as csvfile:
            if not csv.Sniffer().has_header(csvfile.read(2048)):
                build_file = True

        with open(csv_name, 'w+') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["name", "model_class", "arhitecture", "learning_rate", "decay", "batch_size", "epochs",
                                                         "early_stop_patience", "early_stop_min_delta", "reduce_lr_patience", "reduce_lr_factor", "reduce_lr_min_delta", "score", "accuracy", "classification_report", "confusion_matrix"])
            if build_file == True:
                writer.writeheader()
            writer.writerow({"name": self.model_name, "arhitecture": "ensemble", "accuracy": score})
