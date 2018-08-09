import datetime
import itertools
import os

import numpy as np
import pandas as pd
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten, Input
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

from base_model import BaseModel
from keras_vggface.vggface import VGGFace
from utils import load, plot_confusion_matrix, plot_hist


class VGG16Model(BaseModel):

    def imagenet_arhitecture(self):
        model_vgg16_conv = VGG16(include_top=False)
        model_vgg16_conv.summary()

        input = Input(shape=(48, 48, 3), name='image_input')

        output_vgg16_conv = model_vgg16_conv(input)

        x = Flatten(name='flatten')(output_vgg16_conv)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(7, activation='softmax', name='predictions')(x)

        return Model(inputs=input, outputs=x)

    def vggface_arhitecture(self):
        vgg_model = VGGFace(include_top=False, input_shape=(48, 48, 3))
        last_layer = vgg_model.get_layer('pool5').output
        x = Flatten(name='flatten')(last_layer)
        x = Dense(4096, activation='relu', name='fc6')(x)
        x = Dense(4096, activation='relu', name='fc7')(x)
        out = Dense(7, activation='softmax', name='fc8')(x)
        return Model(vgg_model.input, out)

    # train the model and save the weights
    def train(self, X, y, X_validation, y_validation):
        hist = self.model.fit(X, y,
                              shuffle=True,
                              epochs=self.configuration.epochs,
                              batch_size=self.configuration.batch_size,
                              callbacks=self.configuration.callbacks,
                              verbose=1,
                              validation_data=(X_validation, y_validation))

        self.model.save_weights(
            '{0}/model.h5'.format(self.configuration.model_name))

        return hist

    def extract_features(self, layer_name, x, y):
        return self._intermediate_model(layer_name).predict(x)

    def predict(self, x):
        return self.model.predict(x)

    def _evaluate(self, x, y):
        score = self.model.evaluate(x, y, verbose=1)
        predictions = self.predict(x)
        predicted_class = np.argmax(predictions, axis=1)

        return score, predicted_class
