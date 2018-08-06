import datetime
import itertools
import os

import numpy as np
import pandas as pd
from keras.applications.vgg16 import VGG16
from keras.callbacks import (EarlyStopping, ModelCheckpoint,
                             TensorBoard, ReduceLROnPlateau)
from keras.layers import Dense, Flatten, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from skimage.color import gray2rgb
from sklearn.metrics import confusion_matrix, classification_report

MODEL_NAME = "vgg16"


def load(mode=0, normalize_mode=0):
    file = ['fer2013/training.csv',
            'fer2013/publictest.csv',
            'fer2013/privatetest.csv']
    data = pd.read_csv(file[mode])

    pixels = data['pixels'].apply(
        lambda img: np.fromstring(img, sep=' '))

    X = np.vstack(pixels.values)
    X = X.astype('float32')

    if normalize_mode == 0:
        X /= 255
    else:
        X -= np.mean(X, axis=0)
        X /= np.std(X, axis=0)

    X = gray2rgb(X)
    X = X.reshape(-1, 48, 48, 3)

    y = data['emotion'].values
    y = y.astype(np.int)
    y = to_categorical(y)

    return X, y


def model_callbacks(batch_size, early_stop_patience=5, reduce_lr=True, reduce_lr_patience=5, reduce_lr_factor=0.1):
    callbacks = []

    callbacks.append(ModelCheckpoint(filepath='{0}/checkpoint.h5'.format(MODEL_NAME),
                                     verbose=1,
                                     save_best_only=True))
    callbacks.append(EarlyStopping(monitor='val_loss',
                                   min_delta=0,
                                   patience=early_stop_patience,
                                   verbose=2,
                                   mode='auto'))
    callbacks.append(TensorBoard(log_dir='{0}/logs'.format(MODEL_NAME),
                                 histogram_freq=0,
                                 batch_size=batch_size,
                                 write_images=True))

    if reduce_lr:
        callbacks.append(ReduceLROnPlateau(monitor='val_loss',
                                           factor=reduce_lr_factor,
                                           patience=reduce_lr_patience,
                                           verbose=1,
                                           mode='auto'))

    return callbacks


def model_arhitecture(learning_rate, decay):
    model_vgg16_conv = VGG16(include_top=False)
    model_vgg16_conv.summary()

    input = Input(shape=(48, 48, 3), name='image_input')

    output_vgg16_conv = model_vgg16_conv(input)

    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(7, activation='softmax', name='predictions')(x)

    model = Model(inputs=input, outputs=x)
    optimizer = Adam(lr=learning_rate, decay=decay)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(batch_size, epochs, data):
    X, y = load()
    X_validation, y_validation = load(mode=1)
    model = model_arhitecture(0.001, 0.001)
    hist = model.fit(X, y,
                     shuffle=True,
                     epochs=epochs,
                     batch_size=batch_size,
                     callbacks=model_callbacks,
                     verbose=1,
                     validation_data=(X_validation, y_validation))
    model.save_weights('{0}/model.h5'.format(model_name))
