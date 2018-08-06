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


# from comet_ml import Experiment
# experiment = Experiment(api_key="jgPpxpv8cLfz2tpfSneWTftwB")
#  sequence for batch training
#
class Restnet50Sequence(Sequence):
    def __init__(self, x_set, y_set, batch_size=16):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([
            resize(img, (200, 200))
            for img in batch_x]), np.array(batch_y)


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


def load(mode=0, resize_imgs=False):
    file = ['fer2013/training.csv',
            'fer2013/publictest.csv',
            'fer2013/privatetest.csv']
    data = pd.read_csv(file[mode])

    pixels = data['pixels'].apply(
        lambda img: np.fromstring(img, sep=' '))

    X = np.vstack(pixels.values)
    X = X.astype('float32')

    X /= 255

    X = gray2rgb(X)
    X = X.reshape(-1, 48, 48, 3)

    if resize_imgs:
        X = np.array([resize(x, (200, 200)) for x in X])

    y = data['emotion'].values
    y = y.astype(np.int)
    y = to_categorical(y)

    return X, y


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
# import pdb; pdb.set_trace()
clf.fit(features_train, y_train)
score = clf.score(features_test, y_test)
# print(features.shape)
# print(features[0].shape)
# dec = clf.decision_function([[512]])
# print(dec)
print(score)
print(X_test.shape)
print(X_test[1].shape)
dec = clf.decision_function(features_test[1])
print(dec.shape)
