import importlib
import os

from neurst.data.data_pipelines.data_pipeline import DataPipeline
from neurst.utils.registry import setup_registry

build_data_pipeline, register_data_pipeline = setup_registry(DataPipeline.REGISTRY_NAME, base_class=DataPipeline)

models_dir = os.path.dirname(__file__)
for file in os.listdir(models_dir):
    path = os.path.join(models_dir, file)
    if not file.startswith('_') and not file.startswith('.') and file.endswith('.py'):
        model_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module('neurst.data.data_pipelines.' + model_name)
