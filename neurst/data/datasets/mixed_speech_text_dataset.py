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
import yaml
from absl import logging

from neurst.data.datasets import Dataset, build_dataset, register_dataset
from neurst.data.datasets.audio.audio_dataset import AudioTFRecordDataset
from neurst.data.datasets.data_sampler import DataSampler
from neurst.data.datasets.mixed_train_dataset import MixedTrainDataset
from neurst.utils.flags_core import Flag, ModuleFlag


@register_dataset
class MixedSpeechTextTrainRecordDataset(Dataset):
    """ Mixed datasets for training. """

    def __init__(self, args):
        """ Initializes the dataset. """
        super(MixedSpeechTextTrainRecordDataset, self).__init__()
        logging.info("MixedSpeechTextTrainDataset: "
                     "we assume all audio-text datasets are all processed or not.")
        self._args = args

    @staticmethod
    def class_or_method_args():
        return [
            # ASR
            Flag("asr_record_paths", dtype=Flag.TYPE.STRING, default=None,
                 help="A dict of record paths for ASR. The key is the dataset name while "
                      "the value is the path to one ASR dataset (preprocessed TFRecords)."),
            Flag("asr_data_class", dtype=Flag.TYPE.STRING, default=AudioTFRecordDataset.__name__,
                 help="The dataset name for ASR datasets. We assume all ASR datasets have the same format."),
            Flag("asr_common_properties", dtype=Flag.TYPE.STRING, default=None,
                 help="Other common properties for building the ASR record dataset."),
            ModuleFlag("asr_" + DataSampler.REGISTRY_NAME, DataSampler.REGISTRY_NAME, default=None,
                       help="The dataset sampler for unbalanced ASR datasets."),
            # ST
            Flag("st_record_paths", dtype=Flag.TYPE.STRING, default=None,
                 help="A dict of record paths for direct ST. The key is the dataset name while "
                      "the value is the path to one ST dataset (preprocessed TFRecords)."),
            Flag("st_data_class", dtype=Flag.TYPE.STRING, default=AudioTFRecordDataset.__name__,
                 help="The dataset name for ST datasets. We assume all ST datasets have the same format."),
            Flag("st_common_properties", dtype=Flag.TYPE.STRING, default=None,
                 help="Other common properties for building the ST record dataset."),
            ModuleFlag("st_" + DataSampler.REGISTRY_NAME, DataSampler.REGISTRY_NAME, default=None,
                       help="The dataset sampler for unbalanced ST datasets."),
        ]

    def _as_dataset(self, asr_or_st):
        record_paths = self._args[asr_or_st + "_record_paths"]
        data_cls = self._args[asr_or_st + "_data_class"]
        common_properties = self._args[asr_or_st + "_common_properties"]

        if record_paths is None:
            raise ValueError(asr_or_st.upper() + " dataset is not provided.")
        elif isinstance(record_paths, str):
            record_paths = yaml.load(record_paths, Loader=yaml.FullLoader)
        if isinstance(record_paths, str):
            record_paths = {"dataset": record_paths}
        assert isinstance(record_paths, dict)
        if common_properties is None:
            common_properties = {}
        elif isinstance(common_properties, str):
            common_properties = yaml.load(common_properties, Loader=yaml.FullLoader)
        assert isinstance(common_properties, dict)
        if len(record_paths) == 1:
            return build_dataset(data_cls, data_path=list(record_paths.values())[0], **common_properties)
        return build_dataset(
            MixedTrainDataset, data_files={k: {"data_path": v} for k, v in record_paths.items()},
            data_class=data_cls, common_properties=common_properties)

    def as_asr_dataset(self):
        return self._as_dataset("asr")

    def as_st_dataset(self):
        return self._as_dataset("st")

    @property
    def status(self):
        raise NotImplementedError

    def build(self, auto_shard=False, map_func=None, map_output_dtypes=None,
              shuffle=True):
        raise NotImplementedError

    def build_iterator(self, map_func=None, shard_id=0, total_shards=1):
        raise NotImplementedError
