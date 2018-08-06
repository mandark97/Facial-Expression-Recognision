import datetime
import itertools
import os

import numpy as np
import pandas as pd
from keras.callbacks import (EarlyStopping, ModelCheckpoint,
                             TensorBoard, ReduceLROnPlateau)
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras_vggface.vggface import VGGFace
from matplotlib import pyplot as plt
from skimage.color import gray2rgb
from sklearn.metrics import confusion_matrix, classification_report

# Constants

BATCH_SIZE = 256
EPOCHS = 200
PATIENCE = 20
LEARNING_RATE = 0.0001
DELTA = 0.00001
MODEL_NAME = 'vggface_vgg16_v2_255'

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


# model callbacks


checkpointer = ModelCheckpoint(filepath='{0}/checkpoint.h5'.format(MODEL_NAME), verbose=1,
                               save_best_only=True)
early_stop = EarlyStopping(
    monitor='val_loss', min_delta=0, patience=PATIENCE, verbose=2, mode='auto')
tensorboard = TensorBoard(log_dir='{0}/logs'.format(MODEL_NAME), histogram_freq=0, batch_size=BATCH_SIZE,
                          write_images=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto')

model_callbacks = [checkpointer, early_stop, tensorboard, reduce_lr]

# build model

vgg_model = VGGFace(include_top=False, input_shape=(48, 48, 3))
last_layer = vgg_model.get_layer('pool5').output
x = Flatten(name='flatten')(last_layer)
x = Dense(4096, activation='relu', name='fc6')(x)
x = Dense(4096, activation='relu', name='fc7')(x)
out = Dense(7, activation='softmax', name='fc8')(x)
model = Model(vgg_model.input, out)

optimizer = Adam(lr=LEARNING_RATE, decay=DELTA)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy', metrics=['accuracy'])
datagen = ImageDataGenerator(horizontal_flip=True,
                             rotation_range=20,
                             featurewise_center=True,
                             featurewise_std_normalization=True)
X, y = load()
X_test, y_test = load(mode=1)

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
with open('{0}/arhitecture.txt'.format(MODEL_NAME), 'w') as file:
    file.write(model.to_json())

with open('{0}/file.txt'.format(MODEL_NAME), 'w') as file:
    EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    now = str(datetime.datetime.now())
    file.write(MODEL_NAME + ' ' + now + '\n')
    file.write('batch size: ' + str(BATCH_SIZE) + 'epochs: ' + str(EPOCHS) + ' learning rate: ' + str(
        LEARNING_RATE) + 'delta: ' + str(DELTA))
    model.summary(print_fn=lambda x: file.write(x + '\n'))

    x_eval, y_eval = load(mode=2)

    model.save_weights('{0}/model.h5'.format(MODEL_NAME))
    score = model.evaluate(x_eval, y_eval, verbose=1)
    file.write('Score : {0} \n'.format(score[0]))
    file.write('Accuracy : {0} \n'.format(score[1] * 100))

    predictions = model.predict(x_eval)
    predicted_class = np.argmax(predictions, axis=1)
    true_class = np.argmax(y_eval, axis=1)
    file.write(classification_report(true_class, predicted_class, target_names=EMOTIONS))

    cm = confusion_matrix(true_class, predicted_class)
    plt.clf()
    plot_confusion_matrix(cm, classes=EMOTIONS, normalize=True)
    plt.savefig('{0}/confusion_matrix1.png'.format(MODEL_NAME))

    model.load_weights('{0}/checkpoint.h5'.format(MODEL_NAME))
    score = model.evaluate(x_eval, y_eval, verbose=1)
    file.write('Score : {0} \n'.format(score[0]))
    file.write('Accuracy : {0} \n'.format(score[1] * 100))
    predictions = model.predict(x_eval)
    predicted_class = np.argmax(predictions, axis=1)
    true_class = np.argmax(y_eval, axis=1)
    file.write(classification_report(true_class, predicted_class, target_names=EMOTIONS))

    cm = confusion_matrix(true_class, predicted_class)
    plt.clf()
    plot_confusion_matrix(cm, classes=EMOTIONS, normalize=True)
    plt.savefig('{0}/confusion_matrix2.png'.format(MODEL_NAME))
