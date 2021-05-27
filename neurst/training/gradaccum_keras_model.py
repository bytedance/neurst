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
from tensorflow.python.eager import backprop, context
from tensorflow.python.framework import ops
from tensorflow.python.keras import callbacks as callbacks_module
from tensorflow.python.keras.engine import data_adapter, training, training_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.profiler import trace

from neurst.utils.compat import IS_PREV_TF_2_4_0, is_tf_tensor


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


def _multiply_gradient(gradient, scale):
    """Multiply a (possibly sparse) gradient by the given scale factor."""
    if gradient is None:
        return None
    if isinstance(gradient, ops.IndexedSlices):
        return ops.IndexedSlices(
            gradient.values * tf.convert_to_tensor(scale, dtype=gradient.dtype.base_dtype),
            gradient.indices,

            dense_shape=gradient.dense_shape)
    else:
        return gradient * tf.convert_to_tensor(scale, dtype=gradient.dtype.base_dtype)


def _convert_to_tensor(gradient):
    if is_tf_tensor(gradient):
        return gradient
    return tf.convert_to_tensor(gradient)


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
    def __init__(self, accum_steps, average_accumlated_gradients=True):
        """Initializes the accumulator."""
        self._gradients = []
        self._average_accumlated_gradients = average_accumlated_gradients
        self._mult_factor = 1. / accum_steps

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
                    dtype=gradient.dtype.base_dtype,
                    trainable=False,
                    synchronization=tf.VariableSynchronization.ON_READ,
                    aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
                if gradient is not None else gradient for gradient in gradients])

        for accum_gradient, gradient in zip(self._gradients, gradients):
            if accum_gradient is not None and gradient is not None:
                if self._average_accumlated_gradients:
                    accum_gradient.assign_add(
                        _convert_to_tensor(_multiply_gradient(gradient, self._mult_factor)))
                else:
                    accum_gradient.assign_add(_convert_to_tensor(gradient))

    def reset(self):
        """Resets the accumulated gradients on the current replica."""
        for gradient in self._gradients:
            if gradient is not None:
                gradient.assign(tf.zeros_like(gradient))


class GradAccumKerasModel(tf.keras.Model):
    """ Defines the keras model that supports gradient accumulation. """

    def __init__(self, *args, **kwargs):
        self._clip_value = kwargs.pop("clip_value", None)
        self._clip_norm = kwargs.pop("clip_norm", None)
        if self._clip_value:
            self._clip_value = abs(float(self._clip_value))
            if IS_PREV_TF_2_4_0:
                logging.info(f"Ignore clip_value={self._clip_value} in GradAccumKerasModel. "
                             f"Please ensure it is passed to optimizer when TF>=2.4.")
            else:
                logging.info(f"Clipping gradient to = [-{self._clip_value}, {self._clip_value}] "
                             f"in GradAccumKerasModel")
        elif self._clip_norm:
            if IS_PREV_TF_2_4_0:
                logging.info(f"Ignore clip_norm={self._clip_norm} in GradAccumKerasModel. "
                             f"Please ensure it is passed to optimizer when TF>=2.4.")
            else:
                logging.info(f"Clipping gradient norm to {self._clip_norm} in GradAccumKerasModel.")
        self._freeze_variables = kwargs.pop("freeze_variables", None)
        if self._freeze_variables:
            logging.info(f"Variable names matched the pattern {self._freeze_variables} will be frozen.")
        self._update_cycle = kwargs.pop("update_cycle", 1)
        if isinstance(self._update_cycle, str):
            self._update_cycle = int(self._update_cycle)
        if not self._update_cycle:
            self._update_cycle = 1
        self._grad_accumulator = None
        if self._update_cycle > 1:
            logging.info(f"Accumulating gradients for every {self._update_cycle} steps.")
            self._grad_accumulator = GradientAccumulator(self._update_cycle)
        super(GradAccumKerasModel, self).__init__(*args, **kwargs)
        self.accumulate_function = None

    @property
    def trainable_weights(self):
        if self._freeze_variables:
            return [x for x in super(GradAccumKerasModel, self).trainable_weights
                    if re.search(self._freeze_variables, x.name) is None]
        return super(GradAccumKerasModel, self).trainable_weights

    @property
    def non_trainable_weights(self):
        if self._freeze_variables:
            return super(GradAccumKerasModel, self).non_trainable_weights + [
                x for x in super(GradAccumKerasModel, self).trainable_weights
                if re.search(self._freeze_variables, x.name) is not None]
        return super(GradAccumKerasModel, self).non_trainable_weights

    def train_step(self, data, accum_only=False):
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
          accum_only: Whether only accumulate gradients.

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

        if accum_only and self._update_cycle == 1:
            raise ValueError("`accum_only` only supports `update_cycle` > 1.")

        if IS_PREV_TF_2_4_0:
            from tensorflow.python.distribute import parameter_server_strategy
            from tensorflow.python.keras.mixed_precision.experimental import loss_scale_optimizer as lso
            with tape:
                if isinstance(self.optimizer, lso.LossScaleOptimizer):
                    loss = self.optimizer.get_scaled_loss(loss)
                grads = tape.gradient(loss, self.trainable_variables)
            gradients_to_apply = None
            if self._update_cycle > 1:
                self._grad_accumulator(grads)
                if not accum_only:
                    gradients_to_apply = self._grad_accumulator.gradients
            else:
                gradients_to_apply = grads
            if gradients_to_apply is not None:
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
                    gradients_to_apply = self.optimizer._aggregate_gradients(
                        zip(gradients_to_apply, self.trainable_variables))
                if isinstance(self.optimizer, lso.LossScaleOptimizer):
                    gradients_to_apply = self.optimizer.get_unscaled_gradients(gradients_to_apply)
                if self._clip_value:
                    gradients_to_apply = [tf.clip_by_value(grad, -self._clip_value, self._clip_value)
                                          if grad is not None else grad for grad in gradients_to_apply]
                elif self._clip_norm:
                    gradients_to_apply = [tf.clip_by_norm(grad, self._clip_norm)
                                          if grad is not None else grad for grad in gradients_to_apply]
                if self.trainable_variables:
                    if aggregate_grads_outside_optimizer:
                        self.optimizer.apply_gradients(zip(gradients_to_apply, self.trainable_variables),
                                                       experimental_aggregate_gradients=False)
                    else:
                        self.optimizer.apply_gradients(zip(gradients_to_apply, self.trainable_variables))
                if self._grad_accumulator is not None:
                    self._grad_accumulator.reset()
        else:
            grads_and_vars = self.optimizer._compute_gradients(
                loss, self.trainable_variables, tape=tape)
            grads = [g for g, _ in grads_and_vars]

            if self._update_cycle > 1:
                self._grad_accumulator(grads)
                if not accum_only:
                    self.optimizer.apply_gradients(
                        [(g, v) for g, (_, v) in zip(
                            self._grad_accumulator.gradients, grads_and_vars)])
                    self._grad_accumulator.reset()
            else:
                self.optimizer.apply_gradients(grads_and_vars)

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        ret_msg = {m.name: m.result() for m in self.metrics}
        ret_msg["this_step_loss"] = loss
        return ret_msg

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
                outputs = model.train_step(data)
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
            outputs = training.reduce_per_replica(outputs, self.distribute_strategy,
                                                  reduction='first')
            training.write_scalar_summaries(outputs,
                                            step=model._train_counter)  # pylint: disable=protected-access
            return outputs

        def train_function(iterator):
            outputs = step_function(self, iterator)
            return outputs

        def accum(model, iterator):
            """Runs a single training step."""

            def run_step(data):
                model.train_step(data, accum_only=True)

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

        def accum_function(iterator):
            accum(self, iterator)

        if not self.run_eagerly:
            train_function = tf.function(
                train_function, experimental_relax_shapes=True)
            if self._update_cycle > 1:
                accum_function = tf.function(accum_function, experimental_relax_shapes=True)

        self.train_function = train_function
        if self._update_cycle > 1:
            self.accumulate_function = accum_function
        return self.train_function

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
        if IS_PREV_TF_2_4_0:
            training._keras_api_gauge.get_cell('fit').set(True)
        else:
            training.base_layer.keras_api_gauge.get_cell('fit').set(True)
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
            self.train_function = self.make_train_function()
            self._train_counter.assign(0)
            callbacks.on_train_begin()
            training_logs = None
            # Handle fault-tolerance for multi-worker.
            # TODO(omalleyt): Fix the ordering issues that mean this has to
            # happen after `callbacks.on_train_begin`.
            data_handler._initial_epoch = (  # pylint: disable=protected-access
                self._maybe_load_initial_epoch_from_ckpt(initial_epoch))
            logs = None
            for epoch, iterator in data_handler.enumerate_epochs():
                self.reset_metrics()
                callbacks.on_epoch_begin(epoch)
                with data_handler.catch_stop_iteration():
                    for step in data_handler.steps():
                        if IS_PREV_TF_2_4_0:
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
                                tmp_logs = self.train_function(iterator)
                                if data_handler.should_sync:
                                    context.async_wait()
                                logs = tmp_logs  # No error, now safe to assign to logs.
                                end_step = step + data_handler.step_increment
                                callbacks.on_train_batch_end(end_step, logs)
                        else:
                            with trace.Trace(
                                'train',
                                epoch_num=epoch,
                                step_num=step,
                                batch_size=batch_size,
                                _r=1):
                                callbacks.on_train_batch_begin(step)
                                if self._update_cycle > 1:  # gradient accumulation
                                    for _ in range(self._update_cycle - 1):
                                        self.accumulate_function(iterator)
                                tmp_logs = self.train_function(iterator)
                                if data_handler.should_sync:
                                    context.async_wait()
                                logs = tmp_logs  # No error, now safe to assign to logs.
                                end_step = step + data_handler.step_increment
                                callbacks.on_train_batch_end(end_step, logs)
                                if self.stop_training:
                                    break

                if logs is None:
                    raise ValueError('Expect x to be a non-empty array or dataset.')
                epoch_logs = copy.copy(logs)

                # Run validation.
                if validation_data and self._should_eval(epoch, validation_freq):
                    # Create data_handler for evaluation and cache it.
                    if getattr(self, '_eval_data_handler', None) is None:
                        self._fit_frame = training.tf_inspect.currentframe()
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
                if IS_PREV_TF_2_4_0:
                    del self._fit_frame
            callbacks.on_train_end(logs=training_logs)
            return self.history


if IS_PREV_TF_2_4_0:
    class TF23GradAccumKerasModel(GradAccumKerasModel):
        @training.enable_multi_worker
        def fit(self, *args, **kwargs):
            return super(TF23GradAccumKerasModel, self).fit(*args, **kwargs)
