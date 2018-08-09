import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Flatten, Input
from keras.models import Model
from keras.utils import Sequence, normalize, to_categorical
from skimage.transform import resize

from base_model import BaseModel
from keras_vggface.vggface import VGGFace

# sequence for batch training


class Restnet50Sequence(Sequence):
    def __init__(self, x_set, y_set, batch_size):
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


class ResNet50Model(BaseModel):
    def imagenet_arhitecture(self):
        model_restnet_conv = ResNet50(include_top=False)
        model_restnet_conv.summary()

        input = Input(shape=(200, 200, 3))
        output_restnet_conv = model_restnet_conv(input)
        x = Flatten()(output_restnet_conv)
        x = Dense(7, activation='softmax')(x)

        model = Model(inputs=input, outputs=x)

        return model

    def vggface_arhitecture(self):
        vgg_model = VGGFace(model='resnet50', include_top=False,
                            input_shape=(200, 200, 3))
        last_layer = vgg_model.get_layer('avg_pool').output
        x = Flatten(name='flatten')(last_layer)
        out = Dense(7, activation='softmax', name='classifier')(x)
        return Model(vgg_model.input, out)

    def train(self, X, y, X_validation, y_validaton):
        train_generator = Restnet50Sequence(
            X, y, self.configuration.batch_size)
        validation_generator = Restnet50Sequence(
            X_validation, y_validaton, self.configuration.batch_size)
        hist = self.model.fit_generator(train_generator,
                                        shuffle=True,
                                        epochs=self.configuration.epochs,
                                        callbacks=self.configuration.callbacks, verbose=1, use_multiprocessing=True,
                                        steps_per_epoch=len(train_generator), validation_data=validation_generator)

        self.model.save_weights('{0}/model.h5'.format(self.model_name))

        return hist

    def _evaluate(self, x, y):
        test_generator = Restnet50Sequence(x, y, self.configuration.batch_size)
        score = self.model.evaluate_generator(
            test_generator, use_multiprocessing=True)
        predictions = self.model.predict_generator(
            test_generator, use_multiprocessing=True)
        predicted_class = np.argmax(predictions, axis=1)

        return score, predicted_class
