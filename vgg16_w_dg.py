import datetime
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage.color import gray2rgb
from skimage.transform import resize
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                               TensorBoard)
from tensorflow.python.keras.layers import Dense, Flatten, Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils import Sequence, to_categorical

# Constants

BATCH_SIZE = 16
EPOCHS = 200
PATIENCE = 10
LEARNING_RATE = 0.00001
DELTA = 0
MODEL_NAME = 'vgg16_w_dgv2'

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


# sequence for batch training

class VGG16Sequence(Sequence):
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

# model callbacks


checkpointer = ModelCheckpoint(filepath='{0}/checkpoint.h5'.format(MODEL_NAME), verbose=0,
                               save_best_only=True)
early_stop = EarlyStopping(
    monitor='val_loss', min_delta=0, patience=PATIENCE, verbose=2, mode='auto')
tensorboard = TensorBoard(log_dir='{0}/logs'.format(MODEL_NAME), histogram_freq=0, batch_size=BATCH_SIZE,
                          write_images=True)

model_callbacks = [checkpointer, early_stop, tensorboard]

# build model

model_vgg16_conv = VGG16(include_top=False)
model_vgg16_conv.summary()

input = Input(shape=(48, 48, 3), name='image_input')

output_vgg16_conv = model_vgg16_conv(input)

x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dense(7, activation='softmax', name='predictions')(x)

model = Model(inputs=input, outputs=x)
print(model.summary())
optimizer = Adam(lr=LEARNING_RATE, decay=DELTA)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy', metrics=['accuracy'])
datagen = ImageDataGenerator(horizontal_flip=True,
                             rotation_range=20,
                             featurewise_center=True,
                             featurewise_std_normalization=True)
X, y = load()
datagen.fit(X)
X_test, y_test = load(mode=1)

hist = model.fit_generator(datagen.flow(X, y, batch_size=BATCH_SIZE), shuffle=True, epochs=EPOCHS,
                           callbacks=model_callbacks, verbose=1, use_multiprocessing=True,
                           steps_per_epoch=int(len(X) / BATCH_SIZE), validation_data=(X_test, y_test))

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

    score = model.evaluate(x_eval, y_eval, verbose=1)
    file.write('Score : {0} \n'.format(score[0]))
    file.write('Accuracy : {0} \n'.format(score[1] * 100))
    model.save_weights('{0}/model.h5'.format(MODEL_NAME))

    model.load_weights('{0}/checkpoint.h5'.format(MODEL_NAME))
    score = model.evaluate(x_eval, y_eval, verbose=1)
    file.write('Score : {0} \n'.format(score[0]))
    file.write('Accuracy : {0} \n'.format(score[1] * 100))
