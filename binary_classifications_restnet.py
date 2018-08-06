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
from keras_vggface.vggface import VGGFace

# Constants

BATCH_SIZE = 16
EPOCHS = 15
PATIENCE = 3
LEARNING_RATE = 0.00001
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
# sequence for batch training


class Restnet50Sequence(Sequence):
    def __init__(self, x_set, y_set, batch_size=BATCH_SIZE):
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


def create_model(for_class, model_name, X, y, X_test, y_test, x_eval, y_eval):
    checkpointer = ModelCheckpoint(filepath='{0}/checkpoint.h5'.format(model_name), verbose=1,
                                   save_best_only=True)
    early_stop = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=PATIENCE, verbose=2, mode='auto')

    model_callbacks = [checkpointer, early_stop]

    model_vgg16_conv = VGG16(include_top=False)
    model_vgg16_conv.summary()

    vgg_model = VGGFace(model='resnet50', include_top=False,
                        input_shape=(200, 200, 3))
    print(vgg_model)
    last_layer = vgg_model.get_layer('avg_pool').output
    x = Flatten(name='flatten')(last_layer)
    out = Dense(2, activation='softmax', name='classifier')(x)
    model = Model(vgg_model.input, out)

    optimizer = Adam(lr=LEARNING_RATE, decay=DELTA)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy', metrics=['accuracy'])

    vfunc = np.vectorize(class_func)
    y_train_binary = vfunc(y, for_class)
    class_weights = class_weight.compute_class_weight(
        'balanced', np.unique(y_train_binary), y_train_binary)
    y_train_binary = to_categorical(y_train_binary)
    y_test_binary = to_categorical(vfunc(y_test, for_class))
    class_weights = {i: v for i, v in enumerate(class_weights)}
    train_generator = Restnet50Sequence(X, y_train_binary, BATCH_SIZE)
    test_generator = Restnet50Sequence(X_test, y_test_binary, BATCH_SIZE)

    hist = model.fit_generator(train_generator, shuffle=True, epochs=EPOCHS,
                               callbacks=model_callbacks, verbose=1, use_multiprocessing=True,
                               class_weight=class_weights,
                               steps_per_epoch=len(train_generator), validation_data=test_generator)

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
        eval_generator = Restnet50Sequence(x_eval, y_eval_binary, BATCH_SIZE)
        score = model.evaluate_generator(eval_generator, verbose=1)
        file.write('Score : {0} \n'.format(score[0]))
        file.write('Accuracy : {0} \n'.format(score[1] * 100))

        model.load_weights('{0}/checkpoint.h5'.format(model_name))
        score = model.evaluate_generator(eval_generator, verbose=1)
        file.write('Score : {0} \n'.format(score[0]))
        file.write('Accuracy : {0} \n'.format(score[1] * 100))


if __name__ == "__main__":
    X, y = load()
    X_test, y_test = load(mode=1)
    x_eval, y_eval = load(mode=2)

    cls = int(sys.argv[1])
    model_name = "binary_class_resnet_{0}".format(str(cls))
    if not os.path.exists(model_name):
        os.makedirs(model_name)

    create_model(cls, model_name, X, y, X_test, y_test, x_eval, y_eval)
