import importlib
import os

from neurst.data.datasets.data_sampler.data_sampler import DataSampler
from neurst.utils.registry import setup_registry

build_data_sampler, register_data_sampler = setup_registry(DataSampler.REGISTRY_NAME, base_class=DataSampler,
                                                           verbose_creation=True)
models_dir = os.path.dirname(__file__)
for file in os.listdir(models_dir):
    path = os.path.join(models_dir, file)
    if not file.startswith('_') and not file.startswith('.') and file.endswith('.py'):
        model_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module('neurst.data.datasets.data_sampler.' + model_name)
