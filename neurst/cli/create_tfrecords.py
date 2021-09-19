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
# limitation under the License.
import os
import random

import numpy
import tensorflow as tf
import yaml
from absl import app, logging

import neurst.utils.flags_core as flags_core
from neurst.data.datasets import Dataset, build_dataset
from neurst.tasks import Task, build_task
from neurst.utils.compat import ModeKeys

FLAG_LIST = [
    flags_core.Flag("processor_id", dtype=flags_core.Flag.TYPE.INTEGER, default=None,
                    help="The processor id."),
    flags_core.Flag("num_processors", dtype=flags_core.Flag.TYPE.INTEGER, default=None,
                    help="The number of processors. Must be divisible by `num_output_shards`."),
    flags_core.Flag("num_output_shards", dtype=flags_core.Flag.TYPE.INTEGER, default=None,
                    help="The total number of output shards."),
    flags_core.Flag("output_range_begin", dtype=flags_core.Flag.TYPE.INTEGER, default=None,
                    help="The begin ID of output shard (startswith 0, inclusive)."),
    flags_core.Flag("output_range_end", dtype=flags_core.Flag.TYPE.INTEGER, default=None,
                    help="The end ID of output shard (startswith 0, exclusive)."),
    flags_core.Flag("output_template", dtype=flags_core.Flag.TYPE.STRING, default="train.tfrecords-%5.5d-of-%5.5d",
                    help="The template name of output tfrecords, like train.tfrecords-%5.5d-of-%5.5d."),
    flags_core.Flag("mode", dtype=flags_core.Flag.TYPE.STRING, default=ModeKeys.TRAIN,
                    choices=ModeKeys._fields,  # pylint: disable=protected-access
                    help="The mode to acquire data preprocess method, "
                         "that is, the result TF Record dataset will be used for."),
    flags_core.Flag("progressbar", dtype=flags_core.Flag.TYPE.BOOLEAN, default=None,
                    help="Whether to dispaly the progressbar"),
    flags_core.ModuleFlag(Task.REGISTRY_NAME, help="The binding task for data pre-processing."),
    flags_core.ModuleFlag(Dataset.REGISTRY_NAME, help="The raw dataset."),
    flags_core.Flag("extra_kv", dtype=flags_core.Flag.TYPE.STRING,
                    help="extra kv pairs")
]


def _format_tf_feature(feature, dtype):
    if dtype is str:
        feature = tf.nest.map_structure(lambda _x: _x.encode("utf-8"), feature)
    value = numpy.array(feature).flatten()
    if dtype is int:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    elif dtype is float:
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def main(processor_id, num_processors,
         num_output_shards, output_range_begin, output_range_end,
         output_template, dataset: Dataset, progressbar=False, task=None, extra_kv=None):
    if extra_kv is not None:
        if isinstance(extra_kv, str):
            extra_kv = yaml.load(extra_kv, yaml.FullLoader)
        assert isinstance(extra_kv, dict)
    assert 0 <= output_range_begin < output_range_end <= num_output_shards
    assert 0 <= processor_id < num_processors
    logging.info(f"Shards: {output_range_begin} to {output_range_end}")
    if not tf.io.gfile.exists(os.path.dirname(output_template)):
        tf.io.gfile.makedirs(os.path.dirname(output_template))

    file_paths = [output_template % (s, num_output_shards) for s
                  in range(output_range_begin, output_range_end)]
    tmp_file_paths = [f + ".incomplete" for f in file_paths]
    recordio_writers = [tf.io.TFRecordWriter(_x) for _x in tmp_file_paths]

    map_func = None
    if task is not None:
        map_func = task.get_data_preprocess_fn(ModeKeys.TRAIN, dataset.status)
    this_map_func = map_func

    if extra_kv is not None:
        def new_map_func(data):
            data.update(extra_kv)
            if map_func is not None:
                return map_func(data)
            return data

        this_map_func = new_map_func

    feature_type_dict = None
    i = 0
    if progressbar:
        from tqdm import tqdm
        iterator = tqdm(dataset.build_iterator(map_func=this_map_func, shard_id=processor_id,
                                               total_shards=num_processors)(),
                        total=dataset.num_samples // num_processors)
    else:
        iterator = dataset.build_iterator(
            map_func=this_map_func, shard_id=processor_id, total_shards=num_processors)()
    for example in iterator:  # lazily pre-process
        if feature_type_dict is None:
            feature_type_dict = dict()
            for name, data in example.items():
                data_type = type(numpy.array(data).flatten().tolist()[0])
                assert data_type in [int, float, str, bytes], "Not supported {}".format(data_type)
                feature_type_dict[name] = data_type
        feature_dict = {}
        for name, data in example.items():
            feature_dict[name] = _format_tf_feature(data, feature_type_dict[name])
        write_id = random.randint(0, len(recordio_writers) - 1)
        recordio_writers[write_id].write(
            tf.train.Example(features=tf.train.Features(feature=feature_dict)).SerializeToString())
        if i % 1000 == 0:
            recordio_writers[write_id].flush()
        i += 1
    logging.info(f"Total processed {i} samples.")
    for recordio_writer in recordio_writers:
        recordio_writer.close()
    for tmp_f, f in zip(tmp_file_paths, file_paths):
        tf.io.gfile.rename(tmp_f, f, overwrite=True)
    logging.info("===================== Examine feature types =====================")
    for x in tf.data.TFRecordDataset(file_paths).take(1):
        example = tf.train.Example()
        example.ParseFromString(x.numpy())
        logging.info("{")
        for name in example.features.feature:
            if len(example.features.feature[name].bytes_list.value) > 0:
                logging.info(f"    \"{name}\": bytes (str)")
            elif len(example.features.feature[name].int64_list.value) > 0:
                logging.info(f"    \"{name}\": int64")
            elif len(example.features.feature[name].float_list.value) > 0:
                logging.info(f"    \"{name}\": float32")
        logging.info("}")


def _main(_):
    # define and parse program flags
    arg_parser = flags_core.define_flags(FLAG_LIST, with_config_file=True)
    args, remaining_argv = flags_core.intelligent_parse_flags(FLAG_LIST, arg_parser)
    flags_core.verbose_flags(FLAG_LIST, args, remaining_argv)
    task = build_task(args)
    dataset = build_dataset(args)
    if dataset is None:
        raise ValueError("dataset must be provided.")
    main(processor_id=args["processor_id"],
         num_processors=args["num_processors"],
         num_output_shards=args["num_output_shards"],
         output_range_begin=args["output_range_begin"],
         output_range_end=args["output_range_end"],
         output_template=args["output_template"],
         progressbar=args["progressbar"],
         dataset=dataset, task=task, extra_kv=args["extra_kv"])


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    app.run(_main, argv=["pseudo.py"])
