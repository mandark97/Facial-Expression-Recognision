import datetime
import os
from keras_vggface.vggface import VGGFace
import itertools

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage.color import gray2rgb
from skimage.transform import resize
from keras.applications.vgg16 import VGG16
from keras.callbacks import (EarlyStopping, ModelCheckpoint,
                             TensorBoard)
from keras.layers import Dense, Flatten, Input
from keras.models import Model, load_model
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence, to_categorical, get_file
from sklearn.metrics import confusion_matrix

FILEPATH = 'vggface_vgg16v2/model.h5'


def load(mode=0):
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

    y = data['emotion'].values
    y = y.astype(np.int)
    y = to_categorical(y)

    return X, y


model = load_model('binary_class_0/model.h5')

x_eval, y_eval = load(mode=2)

model.load_weights(FILEPATH)
predictions = model.predict(x_eval)

predicted_class = np.argmax(predictions, axis=1)
true_class = np.argmax(y_eval, axis=1)
EMOTIONS_DICT = {0: 'Angry', 1: 'Disgust', 2: 'Fear',
                 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
# for i in range(len(predictions)):
#     plt.title('Prediction:{0}, {1}'.format(
#         EMOTIONS_DICT[predicted_class[i]], EMOTIONS_DICT[np.argmax(y_eval[i])]))
#     plt.imshow(x_eval[i])

#     plt.show()
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

cm = confusion_matrix(true_class, predicted_class)

print(cm)


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


plot_confusion_matrix(cm, classes=EMOTIONS, normalize=True)
plt.show()
