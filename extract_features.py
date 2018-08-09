from __future__ import print_function

import numpy as np
import pandas as pd
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.utils import Sequence, to_categorical
from keras_vggface.vggface import VGGFace
from skimage.color import gray2rgb
from skimage.transform import resize
from sklearn.svm import SVC

from load_data import LoadData


def extract_features(model, layer_name, model_name, data, mode=0, data_generator=None):
    if mode == 0:
        model.load_weights('{0}/model.h5'.format(model_name))
    elif mode == 1:
        model.load_weights('{0}/checkpoint.h5'.format(model_name))

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)

    if data_generator:
        return intermediate_layer_model.predict_generator(data_generator, use_multiprocessing=True)
    else:
        return intermediate_layer_model.predict(data)


X_train, y_train = LoadData(mode=0, rgb=True).load()
X_test, y_test = LoadData(mode=2, rgb=True).load()

def first_model():
    vgg_model = VGGFace(include_top=False, input_shape=(48, 48, 3))
    last_layer = vgg_model.get_layer('pool5').output
    x = Flatten(name='flatten')(last_layer)
    x = Dense(4096, activation='relu', name='fc6')(x)
    x = Dense(4096, activation='relu', name='fc7')(x)
    out = Dense(7, activation='softmax', name='fc8')(x)
    model = Model(vgg_model.input, out)

    features_train = extract_features(model, 'flatten', 'vggface_vgg16v2', X_train, mode=0)
    features_test = extract_features(model, 'flatten', 'vggface_vgg16v2', X_test, mode=0)

    return features_train, features_test


def second_model():
    vgg_model = VGGFace(model='resnet50', include_top=False,
                        input_shape=(200, 200, 3))
    last_layer = vgg_model.get_layer('avg_pool').output
    x = Flatten(name='flatten')(last_layer)
    out = Dense(7, activation='softmax', name='classifier')(x)
    model = Model(vgg_model.input, out)

    train_generator = Restnet50Sequence(X_train, y_train, 16)

    features_train = extract_features(model, 'flatten', 'vggface_restnet50v2', X_train, mode=0,
                                      data_generator=train_generator)
    features_test = extract_features(model, 'flatten', 'vggface_restnet50v2', X_test, mode=0,
                                     data_generator=train_generator)

    return features_train, features_test


model1_train, model1_test = first_model()

model2_train, model2_test = second_model()

features_train = np.concatenate([model1_train, model2_train], axis=1)
features_test = np.concatenate([model1_test, model2_test], axis=1)

clf = SVC(decision_function_shape='ovo', verbose=1)
clf.fit(features_train, y_train)
score = clf.score(features_test, y_test)

print(score)
print(X_test.shape)
print(X_test[1].shape)
dec = clf.decision_function(features_test[1])
print(dec.shape)
