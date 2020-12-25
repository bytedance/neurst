# Copyright 2020 ByteDance Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import re
import traceback

import tensorflow as tf
from absl import logging
from tensorflow.keras import backend as K
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.eager import backprop, context
from tensorflow.python.framework import ops
from tensorflow.python.keras import callbacks as callbacks_module
from tensorflow.python.keras.engine import data_adapter, training, training_utils
from tensorflow.python.keras.mixed_precision.experimental import loss_scale_optimizer as lso
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.profiler import trace


def _multiply_gradient(gradient, scale):
    """Multiply a (possibly sparse) gradient by the given scale factor."""
    if gradient is None:
        return None
    if isinstance(gradient, ops.IndexedSlices):
        return ops.IndexedSlices(
            gradient.values * tf.convert_to_tensor(scale, dtype=gradient.dtype),
            gradient.indices,
            dense_shape=gradient.dense_shape)
    else:
        return gradient * scale


def _minimum_control_deps(outputs):
    """Returns the minimum control dependencies to ensure step succeeded."""
    if context.executing_eagerly():
        return []  # Control dependencies not needed.
    outputs = tf.nest.flatten(outputs, expand_composites=True)
    for out in outputs:
        # Variables can't be control dependencies.
        if not isinstance(out, tf.Variable):
            return [out]  # Return first Tensor or Op from outputs.
    return []  # No viable Tensor or Op to use for control deps.


class GradientAccumulator(object):
    """Gradient accumulation utility.
        When used with a distribution strategy, the accumulator should be called in a
        replica context. Gradients will be accumulated locally on each replica and
        without synchronization. Users should then call ``.gradients``, scale the
        gradients if required, and pass the result to ``apply_gradients``.
        """

    # We use the ON_READ synchronization policy so that no synchronization is
    # performed on assignment. To get the value, we call .value() which returns the
    # value on the current replica without synchronization.
    def __init__(self, accum_steps):
        """Initializes the accumulator."""
        self._gradients = []
        self._accum_grad_scale = 1. / float(accum_steps)

    @property
    def gradients(self):
        """The accumulated gradients on the current replica."""
        return [gradient.value() if gradient is not None else gradient
                for gradient in self._gradients]

    def __call__(self, gradients):
        """Accumulates :obj:`gradients` on the current replica."""
        if len(self._gradients) == 0:
            self._gradients.extend([
                tf.Variable(
                    tf.zeros_like(gradient),
                    trainable=False,
                    synchronization=tf.VariableSynchronization.ON_READ,
                    aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
                if gradient is not None else gradient for gradient in gradients])

        for accum_gradient, gradient in zip(self._gradients, gradients):
            if accum_gradient is not None and gradient is not None:
                K.update_add(accum_gradient, _multiply_gradient(gradient, self._accum_grad_scale))

    def reset(self):
        """Resets the accumulated gradients on the current replica."""
        for gradient in self._gradients:
            if gradient is not None:
                K.update(gradient, tf.zeros_like(gradient))


class AccumgradKerasModel(tf.keras.Model):
    """ Defines the keras model that supports gradient accumulation. """

    def __init__(self, *args, **kwargs):
        self._update_cycle = int(kwargs.pop("update_cycle", 1))
        if not self._update_cycle:
            self._update_cycle = 1
        self._clip_value = kwargs.pop("clip_value", None)
        self._clip_norm = kwargs.pop("clip_norm", None)
        self._freeze_variables = kwargs.pop("freeze_variables", None)
        super(AccumgradKerasModel, self).__init__(*args, **kwargs)
        if self._clip_value:
            self._clip_value = abs(float(self._clip_value))
            logging.info(f"Clipping gradient to = [-{self._clip_value}, {self._clip_value}]")
        elif self._clip_norm:
            logging.info(f"Clipping gradient norm to {self._clip_norm}")
        if self._update_cycle > 1:
            logging.info(f"Accumulating gradients for every {self._update_cycle} steps.")
            self._grad_accumulator = GradientAccumulator(self._update_cycle)
        if self._freeze_variables:
            logging.info(f"Variable names matched the pattern {self._freeze_variables} will be freezed.")
        self.accumulate_function = None

    @property
    def custom_trainable_weights(self):
        if self._freeze_variables:
            return [x for x in self.trainable_weights
                    if re.search(self._freeze_variables, x.name) is None]
        return self.trainable_weights

    @property
    def custom_trainable_variables(self):
        return self.custom_trainable_weights

    def compute_gradients(self, data):
        """The logic for gradient computation of one training step.

        This method can be overridden to support custom training logic.
        This method is called by `Model.make_train_function`.

        This method should contain the mathemetical logic for one step of training.
        This typically includes the forward pass, loss calculation, backpropagation,
        and metric updates.

        Configuration details for *how* this logic is run (e.g. `tf.function` and
        `tf.distribute.Strategy` settings), should be left to
        `Model.make_train_function`, which can also be overridden.

        Arguments:
          data: A nested structure of `Tensor`s.

        Returns:
          A list of gradients corresponding to `Model.trainable_variables`.

        """
        # These are the only transformations `Model.fit` applies to user-input
        # data when a `tf.data.Dataset` is provided. These utilities will be exposed
        # publicly.
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with backprop.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(
                y, y_pred, sample_weight, regularization_losses=self.losses)

        # For custom training steps, users can just write:
        #   trainable_variables = self.trainable_variables
        #   gradients = tape.gradient(loss, trainable_variables)
        #   self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        # The _minimize call does a few extra steps unnecessary in most cases,
        # such as loss scaling and gradient clipping.
        with tape:
            if isinstance(self.optimizer, lso.LossScaleOptimizer):
                loss = self.optimizer.get_scaled_loss(loss)
            gradients = tape.gradient(loss, self.custom_trainable_variables)

        def _zeros_grads():
            zero_gradients = []
            for gradient in gradients:
                if isinstance(gradient, ops.IndexedSlices):
                    zero_gradients.append(ops.IndexedSlices(
                        tf.zeros_like(gradient.values),
                        gradient.indices,
                        dense_shape=gradient.dense_shape))
                elif gradient is None:
                    zero_gradients.append(None)
                else:
                    zero_gradients.append(tf.zeros_like(gradient))
            return zero_gradients

        gradients = tf.cond(tf.math.reduce_all(tf.math.is_finite(loss)),
                            lambda: gradients, _zeros_grads)

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return gradients, {m.name: m.result() for m in self.metrics}

    def custom_apply_gradients(self, gradients):
        trainable_variables = self.custom_trainable_variables
        # Whether to aggregate gradients outside of optimizer. This requires support
        # of the optimizer and doesn't work with ParameterServerStrategy and
        # CentralStroageStrategy.
        aggregate_grads_outside_optimizer = (
            self.optimizer._HAS_AGGREGATE_GRAD  # pylint: disable=protected-access
            and not isinstance(self.distribute_strategy.extended,
                               parameter_server_strategy.ParameterServerStrategyExtended))
        if aggregate_grads_outside_optimizer:
            # We aggregate gradients before unscaling them, in case a subclass of
            # LossScaleOptimizer all-reduces in fp16. All-reducing in fp16 can only be
            # done on scaled gradients, not unscaled gradients, for numeric stability.
            gradients = self.optimizer._aggregate_gradients(zip(gradients,  # pylint: disable=protected-access
                                                                trainable_variables))
        if isinstance(self.optimizer, lso.LossScaleOptimizer):
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        # gradients = self.optimizer._clip_gradients(gradients)  # pylint: disable=protected-access
        if self._clip_value:
            gradients = [tf.clip_by_value(grad, -self._clip_value, self._clip_value)
                         if grad is not None else None for grad in gradients]
        elif self._clip_norm:
            gradients = [tf.clip_by_norm(grad, self._clip_norm)
                         if grad is not None else None for grad in gradients]
        if trainable_variables:
            if aggregate_grads_outside_optimizer:
                self.optimizer.apply_gradients(zip(gradients, trainable_variables),
                                               experimental_aggregate_gradients=False)
            else:
                self.optimizer.apply_gradients(zip(gradients, trainable_variables))

    def make_train_function(self):
        """Creates a function that executes one step of training.

            This method can be overridden to support custom training logic.
            This method is called by `Model.fit` and `Model.train_on_batch`.

            Typically, this method directly controls `tf.function` and
            `tf.distribute.Strategy` settings, and delegates the actual training
            logic to `Model.train_step`.

            This function is cached the first time `Model.fit` or
            `Model.train_on_batch` is called. The cache is cleared whenever
            `Model.compile` is called.

            Returns:
              Function. The function created by this method should accept a
              `tf.data.Iterator`, and return a `dict` containing values that will
              be passed to `tf.keras.Callbacks.on_train_batch_end`, such as
              `{'loss': 0.2, 'accuracy': 0.7}`.
            """
        if self.train_function is not None:
            return self.train_function

        def step_function(model, iterator):
            """Runs a single training step."""

            def run_step(data):
                gradients, outputs = model.compute_gradients(data)
                if self._update_cycle > 1:
                    model._grad_accumulator(gradients)  # pylint: disable=protected-access
                    gradients = model._grad_accumulator.gradients  # pylint: disable=protected-access
                model.custom_apply_gradients(gradients)
                if model._update_cycle > 1:  # pylint: disable=protected-access
                    model._grad_accumulator.reset()  # pylint: disable=protected-access
                # Ensure counter is updated only if `train_step` succeeds.
                with ops.control_dependencies(_minimum_control_deps(outputs)):
                    model._train_counter.assign_add(1)  # pylint: disable=protected-access
                return outputs

            err_cnt = 0
            while err_cnt < 10:
                try:
                    data = next(iterator)
                    break
                except (StopIteration, tf.errors.OutOfRangeError) as e:
                    raise e
                except tf.errors.OpError as e:
                    tf.print(traceback.format_exc(e))
                    err_cnt += 1
                    if err_cnt >= 10:
                        raise e

            outputs = model.distribute_strategy.run(run_step, args=(data,))
            outputs = training.reduce_per_replica(outputs, self.distribute_strategy)
            training.write_scalar_summaries(outputs, step=model._train_counter)  # pylint: disable=protected-access
            return outputs

        def step_function_accum(model, iterator):
            """Runs a single training step."""

            def run_step(data):
                gradients, outputs = model.compute_gradients(data)
                model._grad_accumulator(gradients)  # pylint: disable=protected-access

            err_cnt = 0
            while err_cnt < 10:
                try:
                    data = next(iterator)
                    break
                except (StopIteration, tf.errors.OutOfRangeError) as e:
                    raise e
                except tf.errors.OpError as e:
                    tf.print(traceback.format_exc(e))
                    err_cnt += 1
                    if err_cnt >= 10:
                        raise e

            model.distribute_strategy.run(run_step, args=(data,))

        def accumulate_function(iterator):
            return step_function_accum(self, iterator)

        def train_function(iterator):
            """Runs a training execution with one step."""
            return step_function(self, iterator)

        if not self.run_eagerly:
            train_function = tf.function(
                train_function, experimental_relax_shapes=True)
            if self._update_cycle > 1:
                accumulate_function = tf.function(
                    accumulate_function, experimental_relax_shapes=True)

        self.train_function = train_function
        if self._update_cycle > 1:
            self.accumulate_function = accumulate_function
        return self.train_function

    @training.enable_multi_worker
    def fit(self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_split=0.,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_batch_size=None,
            validation_freq=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False):
        """ Copy from tf.keras.Model. """
        training._keras_api_gauge.get_cell('fit').set(True)
        # Legacy graph support is contained in `training_v1.Model`.
        version_utils.disallow_legacy_graph('Model', 'fit')
        self._assert_compile_was_called()
        self._check_call_args('fit')
        training._disallow_inside_tf_function('fit')

        if validation_split:
            # Create the validation data using the training data. Only supported for
            # `Tensor` and `NumPy` input.
            (x, y, sample_weight), validation_data = (
                data_adapter.train_validation_split(
                    (x, y, sample_weight), validation_split=validation_split))

        if validation_data:
            val_x, val_y, val_sample_weight = (
                data_adapter.unpack_x_y_sample_weight(validation_data))

        with self.distribute_strategy.scope(), \
             training_utils.RespectCompiledTrainableState(self):
            # Creates a `tf.data.Dataset` and handles batch and epoch iteration.
            data_handler = data_adapter.DataHandler(
                x=x,
                y=y,
                sample_weight=sample_weight,
                batch_size=batch_size,
                steps_per_epoch=steps_per_epoch,
                initial_epoch=initial_epoch,
                epochs=epochs,
                shuffle=shuffle,
                class_weight=class_weight,
                max_queue_size=max_queue_size,
                workers=workers,
                use_multiprocessing=use_multiprocessing,
                model=self,
                steps_per_execution=self._steps_per_execution)

            # Container that configures and calls `tf.keras.Callback`s.
            if not isinstance(callbacks, callbacks_module.CallbackList):
                callbacks = callbacks_module.CallbackList(
                    callbacks,
                    add_history=True,
                    add_progbar=verbose != 0,
                    model=self,
                    verbose=verbose,
                    epochs=epochs,
                    steps=data_handler.inferred_steps)

            self.stop_training = False
            train_function = self.make_train_function()
            self._train_counter.assign(0)
            callbacks.on_train_begin()
            training_logs = None
            # Handle fault-tolerance for multi-worker.
            # TODO(omalleyt): Fix the ordering issues that mean this has to
            # happen after `callbacks.on_train_begin`.
            data_handler._initial_epoch = (  # pylint: disable=protected-access
                self._maybe_load_initial_epoch_from_ckpt(initial_epoch))
            for epoch, iterator in data_handler.enumerate_epochs():
                self.reset_metrics()
                callbacks.on_epoch_begin(epoch)
                with data_handler.catch_stop_iteration():
                    if self._update_cycle > 1:
                        self._grad_accumulator.reset()
                    for step in data_handler.steps():
                        with trace.Trace(
                            'TraceContext',
                            graph_type='train',
                            epoch_num=epoch,
                            step_num=step,
                            batch_size=batch_size):
                            callbacks.on_train_batch_begin(step)
                            if self._update_cycle > 1:
                                for _ in range(self._update_cycle - 1):
                                    self.accumulate_function(iterator)
                            tmp_logs = train_function(iterator)
                            if data_handler.should_sync:
                                context.async_wait()
                            logs = tmp_logs  # No error, now safe to assign to logs.
                            end_step = step + data_handler.step_increment
                            callbacks.on_train_batch_end(end_step, logs)
                epoch_logs = copy.copy(logs)

                # Run validation.
                if validation_data and self._should_eval(epoch, validation_freq):
                    # Create data_handler for evaluation and cache it.
                    if getattr(self, '_eval_data_handler', None) is None:
                        self._eval_data_handler = data_adapter.DataHandler(
                            x=val_x,
                            y=val_y,
                            sample_weight=val_sample_weight,
                            batch_size=validation_batch_size or batch_size,
                            steps_per_epoch=validation_steps,
                            initial_epoch=0,
                            epochs=1,
                            max_queue_size=max_queue_size,
                            workers=workers,
                            use_multiprocessing=use_multiprocessing,
                            model=self,
                            steps_per_execution=self._steps_per_execution)
                    val_logs = self.evaluate(
                        x=val_x,
                        y=val_y,
                        sample_weight=val_sample_weight,
                        batch_size=validation_batch_size or batch_size,
                        steps=validation_steps,
                        callbacks=callbacks,
                        max_queue_size=max_queue_size,
                        workers=workers,
                        use_multiprocessing=use_multiprocessing,
                        return_dict=True)
                    val_logs = {'val_' + name: val for name, val in val_logs.items()}
                    epoch_logs.update(val_logs)

                callbacks.on_epoch_end(epoch, epoch_logs)
                training_logs = epoch_logs
                if self.stop_training:
                    break

            # If eval data_hanlder exists, delete it after all epochs are done.
            if getattr(self, '_eval_data_handler', None) is not None:
                del self._eval_data_handler
            callbacks.on_train_end(logs=training_logs)
            return self.history
