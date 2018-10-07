import json
import time


class ModelConfiguration(object):

    def __init__(self, data):
        self.model_class = data['model_class']
        self.arhitecture = data['arhitecture']
        self.model_name = data['model_name']

        if "train_config" in data:
            self.ensemble = False
            self.model_name = self.model_name + str(time.time())
            train_config = data['train_config']

            self.learning_rate = train_config['learning_rate']
            self.decay = train_config['decay']
            self.batch_size = train_config['batch_size']
            self.epochs = train_config['epochs']

            self.early_stop = True if 'early_stop' in train_config else False
            if self.early_stop:
                self.early_stop_patience = train_config['early_stop'].get(
                    'patience', 0)
                self.early_stop_min_delta = train_config['early_stop'].get(
                    'min_delta', 0)

            self.tensorboard = train_config['tensorboard']

            self.reduce_lr = True if 'reduce_lr' in train_config else False
            if self.reduce_lr:
                self.reduce_lr_patience = train_config['reduce_lr'].get(
                    'patience', 10)
                self.reduce_lr_factor = train_config['reduce_lr'].get(
                    'factor', 0.1)
                self.reduce_lr_min_delta = train_config['reduce_lr'].get(
                    'min_delta', 0.0001)
        else:
            self.ensemble = data['ensemble']

    def model(self):
        return eval(self.model_class)(self)

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)

    @staticmethod
    def from_file(file_name):
        with open(file_name, 'r') as f:
            data = json.load(f)

        return ModelConfiguration(data)

    @staticmethod
    def ensemble_from_file(file_name):
        with open(file_name, 'r') as f:
            data = json.load(f)

        return {
            "model_name": data["model_name"],
            "models": [ModelConfiguration(model_config) for model_config in data["models"]]
        }
