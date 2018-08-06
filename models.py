from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.layers import (Activation, Conv2D, Dense, Dropout,
                                            Flatten, Input, MaxPool2D)
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from load_data import LoadData, Restnet50Sequence


def model1(epochs=27, batch_size=256, learning_rate=0, delta=0, model_callbacks=None):
    model = Sequential()
    model.add(Conv2D(64, 3, input_shape=(48, 48, 1), activation='relu'))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(32, 3, activation='relu'))
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(7))
    model.add(Activation('softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    X, y = LoadData().load()
    X_test, y_test = LoadData(mode=1).load()

    hist = model.fit(X, y, shuffle=True, epochs=epochs, batch_size=batch_size,
                     callbacks=model_callbacks, verbose=1,
                     validation_data=(X_test, y_test))
    return model, hist


def model1_w_dg(epochs=300, batch_size=256, learning_rate=0, delta=0, model_callbacks=None):
    model = Sequential()
    model.add(Conv2D(64, 3, input_shape=(48, 48, 1), activation='relu'))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(32, 3, activation='relu'))
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(7))
    model.add(Activation('softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    datagen = ImageDataGenerator(horizontal_flip=True,
                                 rotation_range=20,
                                 featurewise_center=True,
                                 featurewise_std_normalization=True)
    X, y = LoadData().load()
    datagen.fit(X)

    X_test, y_test = LoadData(mode=1).load()

    hist = model.fit_generator(datagen.flow(X, y, batch_size=batch_size), shuffle=True, epochs=epochs,
                               callbacks=model_callbacks, verbose=1,
                               validation_data=(X_test, y_test))
    return model, hist


def vgg16(epochs=27, batch_size=256, learning_rate=0.0001, delta=0.00001, model_callbacks=None):
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
    optimizer = Adam(lr=learning_rate, decay=delta)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy', metrics=['accuracy'])
    X, y = LoadData(rgb=True).load()
    X_test, y_test = LoadData(mode=1, rgb=True).load()

    hist = model.fit(X, y, shuffle=True, epochs=epochs, batch_size=batch_size,
                     callbacks=model_callbacks, verbose=1,
                     validation_data=(X_test, y_test))
    return model, hist


def restnet50(epochs=27, batch_size=256, learning_rate=0.0001, delta=0.00001, model_callbacks=None,
              data_loader=LoadData):
    model_restnet_conv = ResNet50(include_top=False)
    model_restnet_conv.summary()

    input = Input(shape=(200, 200, 3))

    output_restnet_conv = model_restnet_conv(input)

    x = Flatten()(output_restnet_conv)
    x = Dense(7, activation='softmax')(x)

    model = Model(inputs=input, outputs=x)
    print(model.summary())

    optimizer = Adam(lr=learning_rate, decay=delta)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy', metrics=['accuracy'])

    X, y = data_loader(rgb=True).load()
    X_test, y_test = data_loader(
        mode=1, rgb=True).load()
    train_generator = Restnet50Sequence(X, y, 16)
    hist = model.fit_generator(train_generator, shuffle=True, epochs=epochs,
                               callbacks=model_callbacks, verbose=1, use_multiprocessing=True)

    return model, hist
