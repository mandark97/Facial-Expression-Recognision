import datetime
import itertools
import os

import numpy as np
import pandas as pd
from keras.applications.resnet50 import ResNet50
from keras.callbacks import (EarlyStopping, ModelCheckpoint,
                             TensorBoard, ReduceLROnPlateau)
from keras.layers import Dense, Flatten, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import Sequence, to_categorical
from matplotlib import pyplot as plt
from skimage.color import gray2rgb
from skimage.transform import resize
from sklearn.metrics import classification_report, confusion_matrix

# Constants

BATCH_SIZE = 16
EPOCHS = 30
PATIENCE = 7
LEARNING_RATE = 0.0001
DELTA = 0.00001
MODEL_NAME = 'restnet50_final_sec_255'
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


# model callbacks


checkpointer = ModelCheckpoint(filepath='{0}/checkpoint.h5'.format(MODEL_NAME), verbose=0,
                               save_best_only=True)
early_stop = EarlyStopping(
    monitor='val_loss', min_delta=0, patience=PATIENCE, verbose=2, mode='auto')
tensorboard = TensorBoard(log_dir='{0}/logs'.format(MODEL_NAME), histogram_freq=0, batch_size=BATCH_SIZE,
                          write_images=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='auto')

model_callbacks = [checkpointer, early_stop, tensorboard]

# build model

model_restnet_conv = ResNet50(include_top=False, weights=None)
model_restnet_conv.summary()

input = Input(shape=(200, 200, 3))

output_restnet_conv = model_restnet_conv(input)

x = Flatten()(output_restnet_conv)
x = Dense(7, activation='softmax')(x)

model = Model(inputs=input, outputs=x)
print(model.summary())

optimizer = Adam(lr=LEARNING_RATE, decay=DELTA)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy', metrics=['accuracy'])

X, y = load()
X_test, y_test = load(mode=1)
train_generator = Restnet50Sequence(X, y, BATCH_SIZE)
validation_generator = Restnet50Sequence(X_test, y_test, BATCH_SIZE)
hist = model.fit_generator(train_generator, shuffle=True, epochs=EPOCHS,
                           callbacks=model_callbacks, verbose=1, use_multiprocessing=True,
                           steps_per_epoch=len(train_generator), validation_data=validation_generator)

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

    x_eval, y_eval = load(mode=2, resize_imgs=True)
    eval_generator = Restnet50Sequence(x_eval, y_eval, BATCH_SIZE)
    model.save_weights('{0}/model.h5'.format(MODEL_NAME))
    score = model.evaluate_generator(eval_generator, use_multiprocessing=True)

    file.write('Score : {0} \n'.format(score[0]))
    file.write('Accuracy : {0} \n'.format(score[1] * 100))
    predictions = model.predict(x_eval)
    predicted_class = np.argmax(predictions, axis=1)
    true_class = np.argmax(y_eval, axis=1)
    file.write(classification_report(true_class, predicted_class, target_names=EMOTIONS))

    cm = confusion_matrix(true_class, predicted_class)
    plt.clf()
    plt.subplot(1, 2, 1)
    plot_confusion_matrix(cm, classes=EMOTIONS, normalize=True)

    model.load_weights('{0}/checkpoint.h5'.format(MODEL_NAME))
    score = model.evaluate_generator(eval_generator, use_multiprocessing=True)
    file.write('Score : {0} \n'.format(score[0]))
    file.write('Accuracy : {0} \n'.format(score[1] * 100))
    predictions = model.predict(x_eval)
    predicted_class = np.argmax(predictions, axis=1)
    true_class = np.argmax(y_eval, axis=1)
    file.write(classification_report(true_class, predicted_class, target_names=EMOTIONS))

    cm = confusion_matrix(true_class, predicted_class)
    plt.subplot(1, 2, 2)
    plot_confusion_matrix(cm, classes=EMOTIONS, normalize=True)
    plt.savefig('{0}/confusion_matrix.png'.format(MODEL_NAME))
