import datetime
import os

import numpy as np
import pandas as pd
from comet_ml import Experiment, Optimizer, OptimizationMultipleParams
from keras.callbacks import (EarlyStopping, ModelCheckpoint,
                             TensorBoard)
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras_vggface.vggface import VGGFace
from matplotlib import pyplot as plt
from skimage.color import gray2rgb

# Constants
BATCH_SIZE = 256
EPOCHS = 150
PATIENCE = 5
LEARNING_RATE = 0.0001
DELTA = 0.00001
MODEL_NAME = 'vggface_vgg16bs256'

if not os.path.exists(MODEL_NAME):
        os.makedirs(MODEL_NAME)
# load data function

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


# model callbacks


checkpointer = ModelCheckpoint(filepath='{0}/checkpoint.h5'.format(MODEL_NAME), verbose=0,
                               save_best_only=True)
early_stop = EarlyStopping(
    monitor='val_loss', min_delta=0, patience=PATIENCE, verbose=2, mode='auto')
tensorboard = TensorBoard(log_dir='{0}/logs'.format(MODEL_NAME), histogram_freq=0, batch_size=BATCH_SIZE,
                          write_images=True)

model_callbacks = [early_stop, checkpointer, tensorboard]

vgg_model = VGGFace(include_top=False, input_shape=(48, 48, 3))
last_layer = vgg_model.get_layer('pool5').output
x = Flatten(name='flatten')(last_layer)
x = Dense(4096, activation='relu', name='fc6')(x)
x = Dense(4096, activation='relu', name='fc7')(x)
out = Dense(7, activation='softmax', name='fc8')(x)
model = Model(vgg_model.input, out)



experiment = Experiment(api_key="i9Sew6Jy0Z36IZaUfJuR0cxhT",
                        project_name='facialexpressionrecognition')


print(model.summary())
adam_optimizer = Adam(lr=LEARNING_RATE, decay=DELTA)
# optimizer = SGD(lr=LEARNING_RATE, momentum=0.9, decay=DELTA, nesterov=True)
model.compile(optimizer=adam_optimizer,
            loss='categorical_crossentropy', metrics=['accuracy'])

X, y = load()
X_test, y_test = load(mode=1)

params = {'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'decay': DELTA,
        'optimizer': adam_optimizer
        }
with experiment.train():
    hist = model.fit(X, y, shuffle=True, epochs=EPOCHS, batch_size=BATCH_SIZE,
                callbacks=model_callbacks, verbose=1,
                validation_data=(X_test, y_test))

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
plt.savefig('{0}/plot.png'.format(MODEL_NAME))

# evaluate model

with open('{0}/file.txt'.format(MODEL_NAME), 'w') as file:
    now = str(datetime.datetime.now())
    file.write(MODEL_NAME + ' ' + now + '\n')
    file.write('batch size: ' + str(BATCH_SIZE) + 'epochs: ' + str(EPOCHS) + ' learning rate: ' + str(
        LEARNING_RATE) + 'delta: ' + str(DELTA))
    model.summary(print_fn=lambda x: file.write(x + '\n'))

    x_eval, y_eval = load(mode=2)
    with experiment.test():

        score = model.evaluate(x_eval, y_eval, verbose=1)
        file.write('Score : {0} \n'.format(score[0]))
        file.write('Accuracy : {0} \n'.format(score[1] * 100))
        model.save_weights('{0}/model.h5'.format(MODEL_NAME))

        metrics = {
            'loss': score[0],
            'accuracy': score[1]
        }
        experiment.log_multiple_metrics(metrics)

        model.load_weights('{0}/checkpoint.h5'.format(MODEL_NAME))
        score = model.evaluate(x_eval, y_eval, verbose=1)
        file.write('Score : {0} \n'.format(score[0]))
        file.write('Accuracy : {0} \n'.format(score[1] * 100))

        metrics = {
            'loss': score[0],
            'accuracy': score[1]
        }
        experiment.log_multiple_metrics(metrics)

experiment.log_multiple_params(params)
