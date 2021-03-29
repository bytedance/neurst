import importlib
import os

import tensorflow as tf
import yaml
from absl import logging

from neurst.utils.registry import get_registered_class, setup_registry

OPTIMIZER_REGISTRY_NAME = "optimizer"
build_optimizer, register_optimizer = setup_registry(OPTIMIZER_REGISTRY_NAME, base_class=tf.keras.optimizers.Optimizer,
                                                     verbose_creation=True)

models_dir = os.path.dirname(__file__)
for file in os.listdir(models_dir):
    path = os.path.join(models_dir, file)
    if not file.startswith('_') and not file.startswith('.') and file.endswith('.py'):
        model_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module('neurst.optimizers.' + model_name)

Adam = tf.keras.optimizers.Adam
Adagrad = tf.keras.optimizers.Adagrad
Adadelta = tf.keras.optimizers.Adadelta
SGD = tf.keras.optimizers.SGD
register_optimizer(Adam)
register_optimizer(Adagrad)
register_optimizer(Adadelta)
register_optimizer(SGD)


def controlling_optimizer(optimizer, controller, controller_args):
    """ Wrap the optimizer with controller. """
    controller_cls = get_registered_class(controller, OPTIMIZER_REGISTRY_NAME)
    if controller_cls is None:
        return optimizer
    logging.info(f"Wrapper optimizer with controller {controller_cls}")
    new_cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
                   dict(controller_cls.__dict__))
    new_optimizer = new_cls.from_config(optimizer.get_config())
    new_optimizer._HAS_AGGREGATE_GRAD = optimizer._HAS_AGGREGATE_GRAD
    if controller_args is None:
        controller_args = {}
    elif isinstance(controller_args, str):
        controller_args = yaml.load(controller_args, Loader=yaml.FullLoader)
    assert isinstance(controller_args, dict)
    new_optimizer.reset_hparams(controller_args)
    return new_optimizer
