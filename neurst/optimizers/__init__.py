import tensorflow as tf

from neurst.utils.registry import setup_registry

OPTIMIZER_REGISTRY_NAME = "optimizer"
build_optimizer, register_optimizer = setup_registry(OPTIMIZER_REGISTRY_NAME, base_class=tf.keras.optimizers.Optimizer,
                                                     verbose_creation=True)

Adam = tf.keras.optimizers.Adam
Adagrad = tf.keras.optimizers.Adagrad
Adadelta = tf.keras.optimizers.Adadelta
SGD = tf.keras.optimizers.SGD
register_optimizer(Adam)
register_optimizer(Adagrad)
register_optimizer(Adadelta)
register_optimizer(SGD)
