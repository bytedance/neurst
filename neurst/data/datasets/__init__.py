import importlib
import os

from neurst.data.datasets.dataset import Dataset, TFRecordDataset
from neurst.utils.registry import setup_registry

build_dataset, register_dataset = setup_registry(Dataset.REGISTRY_NAME, base_class=Dataset,
                                                 verbose_creation=True)
_ = TFRecordDataset
models_dir = os.path.dirname(__file__)
for file in os.listdir(models_dir):
    path = os.path.join(models_dir, file)
    if not file.startswith('_') and not file.startswith('.') and file.endswith('.py'):
        model_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module('neurst.data.datasets.' + model_name)

importlib.import_module("neurst.data.datasets.audio")
importlib.import_module("neurst.data.datasets.data_sampler")
