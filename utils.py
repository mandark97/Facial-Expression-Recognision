import datetime
import itertools
import os

import numpy as np
import pandas as pd
from keras.applications.vgg16 import VGG16
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)
from keras.layers import Dense, Flatten, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from skimage.color import gray2rgb

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


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


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_hist(hist, model_name):
    plt.figure(figsize=(14, 3))
    plt.subplot(1, 2, 1)
    plt.suptitle('Optimizer : Adam', fontsize=10)
    plt.ylabel('Loss', fontsize=16)
    plt.plot(hist.history['loss'], 'b', label='Training Loss')
    plt.plot(hist.history['val_loss'], 'r', label='Validation Loss')
    plt.legend(loc='upper right')

    plt.subplot(1, 2, 2)
    plt.ylabel('Accuracy', fontsize=16)
    plt.plot(hist.history['acc'], 'b', label='Training Accuracy')
    plt.plot(hist.history['val_acc'], 'r', label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.savefig('{0}/plot.png'.format(model_name))


FILENAME = 'fer2013/fer2013.csv'
FOLDER = 'fer2013'


def format_filename(name):
    filename = '-'.join(name.lower().split())
    return '{0}/{1}.csv'.format(FOLDER, filename)


def separate_data(filename):
    data = pd.read_csv(filename)

    for t in data['Usage'].unique():
        df = data.loc[data['Usage'] == t]

        df = df.drop('Usage', 1)
        df.to_csv(format_filename(t), sep=',', index=False)
