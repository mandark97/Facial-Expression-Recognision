import numpy as np
import pandas as pd
from skimage.color import gray2rgb
from skimage.transform import resize
from tensorflow.python.keras.utils import Sequence, to_categorical

FILENAME = 'fer2013/fer2013.csv'
FOLDER = 'fer2013'


def format_filename(name):
    filename = '-'.join(name.lower().split())
    return '{0}/{1}.csv'.format(FOLDER, filename)


def separate_data(filename):
    data = pd.read_csv(filename)

    for t in data['Usage'].unique():
        df = data.loc[data['Usage'] == t]

        df = df.drop('Usage', 1)
        df.to_csv(format_filename(t), sep=',', index=False)


class LoadData(object):
    def __init__(self, test=False, mode=0, rgb=False, resize=False):
        self.test = test
        self.file = ['fer2013/training.csv',
                     'fer2013/publictest.csv',
                     'fer2013/privatetest.csv']
        self.mode = mode
        self.rgb = rgb
        self.resize = resize

    def send_data(self):
        return pd.read_csv(self.file[self.mode])

    def load(self):
        data = pd.read_csv(self.file[self.mode])

        data['pixels'] = data['pixels'].apply(
            lambda img: np.fromstring(img, sep=' '))
        X = np.vstack(data['pixels'].values)
        # #normalize data
        X = X.astype('float32')

        X /= 255

        if self.rgb:  # transform data cu rgb if needed
            X = gray2rgb(X)
            X = X.reshape(-1, 48, 48, 3)
        else:
            X = X.reshape(-1, 48, 48, 1)

        if not self.test:
            y = data['emotion'].values
            y = y.astype(np.int)
            # y = to_categorical(y)
        else:
            y = None

        return X, y


class LoadDataInBatches(object):
    def __init__(self, test=False, mode=0, rgb=False, batch_size=256):
        self.test = test
        self.file = ['fer2013/training.csv',
                     'fer2013/publictest.csv',
                     'fer2013/privatetest.csv']
        self.mode = mode
        self.rgb = rgb

    def load(self, resize_imgs=False):
        data = pd.read_csv(self.file[self.mode])
        pixels = data['pixels'].apply(
            lambda img: np.fromstring(img, sep=' '))
        X = np.vstack(pixels.values)
        X = X.astype('float32')
        # #normalize data
        # X -= np.mean(X, axis=0)
        # X /= np.std(X, axis=0)
        X /= 255

        X = gray2rgb(X)
        X = X.reshape(-1, 48, 48, 3)
        if resize_imgs:
            X = np.array([resize(x, (200, 200)) for x in X])

        if not self.test:
            y = data['emotion'].values
            y = y.astype(np.int)
            y = to_categorical(y)
        else:
            y = None

        return X, y


# Here, `x_set` is list of path to the images
# and `y_set` are the associated classes.


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
