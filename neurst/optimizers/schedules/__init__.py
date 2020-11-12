import importlib
import os

import tensorflow as tf

from neurst.utils.registry import setup_registry

LR_SCHEDULE_REGISTRY_NAME = "lr_schedule"
build_lr_schedule, register_lr_schedule = setup_registry(
    LR_SCHEDULE_REGISTRY_NAME, base_class=tf.keras.optimizers.schedules.LearningRateSchedule,
    verbose_creation=True)

models_dir = os.path.dirname(__file__)
for file in os.listdir(models_dir):
    path = os.path.join(models_dir, file)
    if not file.startswith('_') and not file.startswith('.') and file.endswith('.py'):
        model_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module('neurst.optimizers.schedules.' + model_name)
