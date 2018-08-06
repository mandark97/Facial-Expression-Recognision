import os

import matplotlib.pyplot as plt

from load_data import LoadData

curdir = os.path.abspath(os.path.dirname(__file__))

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

EMOTIONS_DICT2 = {'0': 'Angry', '1': 'Disgust', '2': 'Fear', '3': 'Happy', '4': 'Sad', '5': 'Surprise',
                  '6': 'Neutral'}
EMOTIONS_DICT = {0: 'Angry', 1: 'Disgust', 2: 'Fear',
                 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}


def show_img(images, labels):
    emotions = labels
    emotions_index = [300, 5000, 7000, 25000]

    images = images.reshape(images.shape[0], 48, 48)

    for i in range(len(emotions_index)):
        plt.subplot(1, 4, i + 1)
        plt.axis('off')
        plt.imshow(images[emotions_index[i]])
        plt.title(emotions[emotions_index[i]])
        plt.subplots_adjust(wspace=0.5)
    plt.show()


if __name__ == "__main__":
    images, labels = LoadData().load()
    show_img(images, labels)
