import os

import models
from load_data import LoadData, LoadDataInBatches
from utils import model_callbacks, plot_model, test_model

BATCH_SIZE = 256
EPOCHS = 27
PATIENCE = 50
LEARNING_RATE = 0.01
DELTA = 'default'
MODEL_NAME = 'restnet50'


if __name__ == "__main__":
    if not os.path.exists(MODEL_NAME):
        os.makedirs(MODEL_NAME)

    opts = {
        'model_name': MODEL_NAME,
        'epochs': EPOCHS,
        'patience': PATIENCE,
        'learning_rate': LEARNING_RATE,
        'delta': DELTA,
        'batch_size': BATCH_SIZE,
    }
    model, hist = models.vgg16(model_callbacks=model_callbacks(opts))
    plot_model(hist, MODEL_NAME)
    test_model(model, LoadData, opts)
