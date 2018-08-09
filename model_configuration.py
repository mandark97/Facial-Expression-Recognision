import json


class ModelConfiguration(object):

    def __init__(self, data):
        self.name = data['name']
        self.arhitecture = data['arhitecture']

        self.learning_rate = data['learning_rate']
        self.decay = data['decay']
        self.batch_size = data['batch_size']
        self.epochs = data['epochs']

        self.early_stop = True if 'early_stop' in data else False
        if self.early_stop:
            self.early_stop_patience = data['early_stop'].get(
                'patience', 0)
            self.early_stop_min_delta = data['early_stop'].get('min_delta', 0)

        self.tensorboard = data['tensorboard']

        self.reduce_lr = True if 'reduce_lr' in data else False
        if self.reduce_lr:
            self.reduce_lr_factor = data['reduce_lr'].get('factor', 0.1)
            self.reduce_lr_patience = data['reduce_lr'].get('patience', 10)
            self.reduce_lr_min_delta = data['reduce_lr'].get(
                'min_delta', 0.0001)

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)

    @staticmethod
    def from_file(file_name):
        with open(file_name, 'r') as f:
            data = json.load(f)

        return ModelConfiguration(data)
