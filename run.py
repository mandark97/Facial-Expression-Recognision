from trainer import Trainer
from model_configuration import ModelConfiguration
from vgg16_model import VGG16Model
from restnet50_model import ResNet50Model

trainer = Trainer(ModelConfiguration.from_file('configs/model_config_example.json'))
trainer.train()
trainer.evaluate()
