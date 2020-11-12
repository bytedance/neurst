import importlib
import os

from neurst.training.callbacks import (CentralizedCallback, CustomCheckpointCallback, LearningRateScheduler,
                                       MetricReductionCallback)
from neurst.training.validator import Validator
from neurst.utils.registry import setup_registry

build_validator, register_validator = setup_registry(Validator.REGISTRY_NAME, base_class=Validator,
                                                     verbose_creation=True)

__all__ = [
    "CentralizedCallback",
    "CustomCheckpointCallback",
    "LearningRateScheduler",
    "MetricReductionCallback",

    "Validator",
    "register_validator",
    "build_validator"
]

models_dir = os.path.dirname(__file__)
for file in os.listdir(models_dir):
    path = os.path.join(models_dir, file)
    if not file.startswith('_') and not file.startswith('.') and file.endswith('.py'):
        model_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module('neurst.training.' + model_name)
