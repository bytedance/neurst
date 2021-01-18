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
""" Defines utils for distributed training. """
import json
import os

import tensorflow as tf

from neurst.utils.compat import IS_PREV_TF_2_4_0, register_distributed_worker_setting


def tpu_initialize(tpu_address):
    """Initializes TPU for TF 2.0 training.

    Args:
      tpu_address: string, bns address of master TPU worker.

    Returns:
      A TPUClusterResolver.
    """
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu=tpu_address)
    if tpu_address not in ("", "local"):
        tf.config.experimental_connect_to_cluster(cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    return cluster_resolver


def _collective_communication(all_reduce_alg):
    """Return a CollectiveCommunication based on all_reduce_alg.

    Args:
      all_reduce_alg: a string specifying which collective communication to pick,
        or None.

    Returns:
      tf.distribute.experimental.CollectiveCommunication object

    Raises:
      ValueError: if `all_reduce_alg` not in [None, 'ring', 'nccl']
    """
    if IS_PREV_TF_2_4_0:
        collective_communication_options = {
            None: tf.distribute.experimental.CollectiveCommunication.AUTO,
            "ring": tf.distribute.experimental.CollectiveCommunication.RING,
            "nccl": tf.distribute.experimental.CollectiveCommunication.NCCL
        }
    else:
        collective_communication_options = {
            None: tf.distribute.experimental.CommunicationImplementation.AUTO,
            "ring": tf.distribute.experimental.CommunicationImplementation.RING,
            "nccl": tf.distribute.experimental.CommunicationImplementation.NCCL
        }
    if all_reduce_alg not in collective_communication_options:
        raise ValueError(
            "When used with `multi_worker_mirrored`, valid values for "
            "all_reduce_alg are ['ring', 'nccl'].  Supplied value: {}".format(
                all_reduce_alg))
    return collective_communication_options[all_reduce_alg]


def _mirrored_cross_device_ops(all_reduce_alg, num_packs):
    """Return a CrossDeviceOps based on all_reduce_alg and num_packs.

    Args:
      all_reduce_alg: a string specifying which cross device op to pick, or None.
      num_packs: an integer specifying number of packs for the cross device op.

    Returns:
      tf.distribute.CrossDeviceOps object or None.

    Raises:
      ValueError: if `all_reduce_alg` not in [None, 'nccl', 'hierarchical_copy'].
    """
    if all_reduce_alg is None:
        return None
    mirrored_all_reduce_options = {
        "nccl": tf.distribute.NcclAllReduce,
        "hierarchical_copy": tf.distribute.HierarchicalCopyAllReduce
    }
    if all_reduce_alg not in mirrored_all_reduce_options:
        raise ValueError(
            "When used with `mirrored`, valid values for all_reduce_alg are "
            "['nccl', 'hierarchical_copy'].  Supplied value: {}".format(
                all_reduce_alg))
    cross_device_ops_class = mirrored_all_reduce_options[all_reduce_alg]
    return cross_device_ops_class(num_packs=num_packs)


def get_distribution_strategy(distribution_strategy="mirrored",
                              num_gpus=0,
                              worker_hosts=None,
                              task_index=-1,
                              all_reduce_alg="nccl",
                              num_packs=1,
                              tpu_address=None):
    """Return a DistributionStrategy for running the model.

    Args:
      distribution_strategy: a string specifying which distribution strategy to
        use. Accepted values are 'off', 'default', 'one_device', 'mirrored',
        'parameter_server', 'multi_worker_mirrored', and 'tpu' -- case insensitive.
        'off' means not to use Distribution Strategy; 'default' means to choose from
        `MirroredStrategy`, `MultiWorkerMirroredStrategy`, or `OneDeviceStrategy`
        according to the number of GPUs and number of workers. 'tpu' means to use
        TPUStrategy using `tpu_address`.
      num_gpus: Number of GPUs to run this model.
      worker_hosts: The worker hosts for 'multi_worker_mirrored'.
      task_index: The task index for 'multi_worker_mirrored'.
      all_reduce_alg: Optional. Specifies which algorithm to use when performing
        all-reduce. For `MirroredStrategy`, valid values are "nccl" and
        "hierarchical_copy". For `MultiWorkerMirroredStrategy`, valid values are
        "ring" and "nccl".  If None, DistributionStrategy will choose based on
        device topology.
      num_packs: Optional.  Sets the `num_packs` in `tf.distribute.NcclAllReduce`
        or `tf.distribute.HierarchicalCopyAllReduce` for `MirroredStrategy`.
      tpu_address: Optional. String that represents TPU to connect to. Must not
        be None if `distribution_strategy` is set to `tpu`.
    Returns:
      tf.distribute.DistibutionStrategy object.
    Raises:
      ValueError: if `distribution_strategy` is 'off' or 'one_device' and
        `num_gpus` is larger than 1; or `num_gpus` is negative or if
        `distribution_strategy` is `tpu` but `tpu_address` is not specified.
    """
    if num_gpus == 0:
        num_gpus = int(os.environ.get("WORKER_GPUS", '0'))
    if num_gpus < 0:
        raise ValueError("`num_gpus` can not be negative.")
    if (distribution_strategy is None or distribution_strategy.lower() == "none"
        or distribution_strategy == ""):
        return None
    distribution_strategy = distribution_strategy.lower()

    if distribution_strategy == "tpu":
        # When tpu_address is an empty string, we communicate with local TPUs.
        cluster_resolver = tpu_initialize(tpu_address)
        return tf.distribute.TPUStrategy(cluster_resolver)

    if distribution_strategy == "multi_worker_mirrored":
        if worker_hosts is None:
            worker_hosts = os.environ.get("WORKER_HOSTS", None)
            task_index = int(os.environ.get("TASK_ID", -1))
        assert worker_hosts, (
            "worker_hosts must be provided when using 'multi_worker_mirrored'.")

        workers = worker_hosts.split(',')
        if len(workers) > 1 and 0 > task_index:
            raise ValueError('Must specify task_index when number of workers > 1')
        task_index = 0 if len(workers) == 1 else task_index
        register_distributed_worker_setting(worker_id=task_index, num_workers=len(workers),
                                            strategy="multi_worker_mirrored")
        os.environ['TF_CONFIG'] = json.dumps({
            'cluster': {
                'worker': workers
            },
            'task': {'type': 'worker', 'index': task_index}
        })
        # if IS_TF_2_3: # TODO fit non-experimental multiworker strategy
        strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
            communication=_collective_communication(all_reduce_alg))
        strategy.extended.experimental_enable_get_next_as_optional = True
        return strategy

    if distribution_strategy in ("mirrored", "default"):
        return tf.distribute.MirroredStrategy(
            cross_device_ops=_mirrored_cross_device_ops(
                all_reduce_alg, num_packs))

    raise ValueError(
        "Unrecognized Distribution Strategy: %r" % distribution_strategy)
