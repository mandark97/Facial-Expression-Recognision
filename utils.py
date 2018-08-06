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


def evaluate_model(model, model_name, training_data, evaluate_function):
    # evaluate model
    with open('{0}/arhitecture.txt'.format(model_name), 'w') as file:
        file.write(model.to_json())

    with open('{0}/file.txt'.format(model_name), 'w') as file:
        EMOTIONS = ['Angry', 'Disgust', 'Fear',
                    'Happy', 'Sad', 'Surprise', 'Neutral']

        now = str(datetime.datetime.now())
        file.write(model_name + ' ' + now + '\n')
        file.write('batch size: ' + str(training_data['batch_size']) + 'epochs: ' + str(training_data['epochs']) + ' learning rate: ' + str(
            training_data['learning_rate']) + 'decay: ' + str(training_data['decay']))
        model.summary(print_fn=lambda x: file.write(x + '\n'))

        x_eval, y_eval = load(mode=2)

        model.save_weights('{0}/model.h5'.format(model_name))
        score = model.evaluate(x_eval, y_eval, verbose=1)
        file.write('Score : {0} \n'.format(score[0]))
        file.write('Accuracy : {0} \n'.format(score[1] * 100))

        predictions = model.predict(x_eval)
        predicted_class = np.argmax(predictions, axis=1)
        true_class = np.argmax(y_eval, axis=1)
        file.write(classification_report(
            true_class, predicted_class, target_names=EMOTIONS))

        cm = confusion_matrix(true_class, predicted_class)
        plt.clf()
        plt.subplot(1, 2, 1)
        plot_confusion_matrix(cm, classes=EMOTIONS, normalize=True)

        model.load_weights('{0}/checkpoint.h5'.format(model_name))
        score = model.evaluate(x_eval, y_eval, verbose=1)
        file.write('Score : {0} \n'.format(score[0]))
        file.write('Accuracy : {0} \n'.format(score[1] * 100))
        predictions = model.predict(x_eval)
        predicted_class = np.argmax(predictions, axis=1)
        true_class = np.argmax(y_eval, axis=1)
        file.write(classification_report(
            true_class, predicted_class, target_names=EMOTIONS))

        cm = confusion_matrix(true_class, predicted_class)
        plt.subplot(1, 2, 2)
        plot_confusion_matrix(cm, classes=EMOTIONS, normalize=True)
        plt.savefig('{0}/confusion_matrix.png'.format(model_name))
