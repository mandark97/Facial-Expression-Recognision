import datetime
import os
from abc import ABC, abstractmethod

import numpy as np
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from utils import plot_confusion_matrix

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


class BaseModel(ABC):
    def __init__(self, model_configuration):
        self.configuration = model_configuration
        self.model_name = self.configuration.model_name
        if not os.path.exists(self.model_name):
            os.makedirs(self.model_name)

        if self.configuration.arhitecture == 'imagenet':
            self.model = self.imagenet_arhitecture()
        elif self.configuration.arhitecture == 'vggface':
            self.model = self.vggface_arhitecture()

        self._compile()
        self.callbacks = self._model_callbacks()

    def evaluate_model(self, x_test, y_test):
        self._save_arhitecture()

        report = self._reports(x_test, y_test)

        self._save_to_file(report)
        self._save_confusion_matrix(report)

    @abstractmethod
    def imagenet_arhitecture(self):
        pass

    @abstractmethod
    def vggface_arhitecture(self):
        pass

    @abstractmethod
    def train(self, X, y, X_validation, y_validaton):
        pass

    @abstractmethod
    def _evaluate(self, x, y):
        pass

    def _model_callbacks(self):
        callbacks = []

        callbacks.append(ModelCheckpoint(filepath='{0}/checkpoint.h5'.format(self.model_name),
                                         verbose=1,
                                         save_best_only=True))

        if self.configuration.early_stop:
            callbacks.append(EarlyStopping(monitor='val_loss',
                                           min_delta=0,
                                           patience=self.configuration.early_stop_patience,
                                           verbose=2,
                                           mode='auto'))
        if self.configuration.tensorboard:
            callbacks.append(TensorBoard(log_dir='{0}/logs'.format(self.model_name),
                                         histogram_freq=0,
                                         batch_size=self.configuration.batch_size,
                                         write_images=True))

        if self.configuration.reduce_lr:
            callbacks.append(ReduceLROnPlateau(monitor='val_loss',
                                               factor=self.configuration.reduce_lr_factor,
                                               patience=self.configuration.reduce_lr_patience,
                                               verbose=1,
                                               mode='auto'))

        return callbacks

    def _compile(self):
        optimizer = Adam(lr=self.configuration.learning_rate,
                         decay=self.configuration.decay)

        self.model.compile(optimizer=optimizer,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def _reports(self, x, y):
        report = {}
        report['score'], predicted_class = self._evaluate(x, y)
        true_class = np.argmax(y, axis=1)
        report['classication_report'] = classification_report(
            true_class, predicted_class, target_names=EMOTIONS)
        report['confusion_matrix'] = confusion_matrix(
            true_class, predicted_class)

        return report

    def _save_arhitecture(self):
        with open('{0}/arhitecture.txt'.format(self.model_name), 'w') as file:
            file.write(self.model.to_json())

    def _save_to_file(self, report):
        with open('{0}/file.txt'.format(self.model_name), 'w') as file:
            now = str(datetime.datetime.now())
            file.write(self.model_name + ' ' + now + '\n')
            file.write('batch size: ' + str(self.configuration.batch_size)
                       + ' epochs: ' + str(self.configuration.epochs)
                       + ' learning rate: ' +
                       str(self.configuration.learning_rate)
                       + ' decay: ' + str(self.configuration.decay))

            self.model.summary(print_fn=lambda x: file.write(x + '\n'))

            file.write('Score : {0} \n'.format(report['score'][0]))
            file.write('Accuracy : {0} \n'.format(
                report['score'][1] * 100))
            file.write(report['classification_report'])

    def _save_confusion_matrix(self, report):
        plt.clf()
        plot_confusion_matrix(
            report['confusion_matrix'], classes=EMOTIONS, normalize=True)
        plt.savefig('{0}/confusion_matrix.png'.format(self.model_name))
