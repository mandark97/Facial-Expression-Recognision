import datetime
import os
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage.color import gray2rgb
from skimage.transform import resize
from keras.applications.vgg16 import VGG16
from keras.callbacks import (EarlyStopping, ModelCheckpoint,
                             TensorBoard)
from keras.layers import Dense, Flatten, Input
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence, to_categorical
from sklearn.utils import class_weight

# Constants

BATCH_SIZE = 256
EPOCHS = 100
PATIENCE = 10
LEARNING_RATE = 0.0001
DELTA = 0.00001
MODEL_NAME = 'binary_classification'


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

    return X, y


def class_func(value, good_class):
    if value == good_class:
        return 1
    else:
        return 0

def create_model(for_class, model_name, X, y, X_test, y_test, x_eval, y_eval):
    checkpointer = ModelCheckpoint(filepath='{0}/checkpoint.h5'.format(model_name), verbose=1,
                                save_best_only=True)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=PATIENCE, verbose=2, mode='auto')

    model_vgg16_conv = VGG16(include_top=False)
    model_vgg16_conv.summary()

    input = Input(shape=(48, 48, 3), name='image_input')

    output_vgg16_conv = model_vgg16_conv(input)

    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(2, activation='softmax', name='predictions')(x)

    model = Model(inputs=input, outputs=x)

    optimizer = Adam(lr=LEARNING_RATE, decay=DELTA)
    model.compile(optimizer=optimizer,
                loss='categorical_crossentropy', metrics=['accuracy'])

    vfunc = np.vectorize(class_func)
    y_train_binary = vfunc(y, for_class)
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train_binary), y_train_binary)
    y_train_binary = to_categorical(y_train_binary)
    y_test_binary = to_categorical(vfunc(y_test, for_class))
    class_weights = { i: v for i, v in enumerate(class_weights) }
    hist = model.fit(X, y_train_binary, shuffle=True, epochs=EPOCHS, batch_size=BATCH_SIZE,
                     callbacks=[early_stop, checkpointer], verbose=1,
                    class_weight=class_weights, validation_data=(X_test, y_test_binary))

    # plot results

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

    # evaluate model

    with open('{0}/file.txt'.format(model_name), 'w') as file:
        now = str(datetime.datetime.now())
        file.write(model_name + ' ' + now + '\n')
        file.write('batch size: ' + str(BATCH_SIZE) + 'epochs: ' + str(EPOCHS) + ' learning rate: ' + str(
            LEARNING_RATE) + 'delta: ' + str(DELTA))
        model.summary(print_fn=lambda x: file.write(x + '\n'))
        model.save_weights('{0}/model.h5'.format(model_name))

        y_eval_binary = to_categorical(vfunc(y_eval, for_class))

        score = model.evaluate(x_eval, y_eval_binary, verbose=1)
        file.write('Score : {0} \n'.format(score[0]))
        file.write('Accuracy : {0} \n'.format(score[1] * 100))

        model.load_weights('{0}/checkpoint.h5'.format(model_name))
        score = model.evaluate(x_eval, y_eval_binary, verbose=1)
        file.write('Score : {0} \n'.format(score[0]))
        file.write('Accuracy : {0} \n'.format(score[1] * 100))

if __name__ == "__main__":
    X, y = load()
    X_test, y_test = load(mode=1)
    x_eval, y_eval = load(mode=2)
    cls = int(sys.argv[1])
    model_name = "binary_class_{0}".format(str(cls))
    if not os.path.exists(model_name):
        os.makedirs(model_name)

    create_model(cls, model_name, X, y, X_test, y_test, x_eval, y_eval)
