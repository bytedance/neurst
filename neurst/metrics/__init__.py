import importlib
import os

from neurst.metrics.metric import Metric
from neurst.utils.registry import setup_registry

build_metric, register_metric = setup_registry(Metric.REGISTRY_NAME, base_class=Metric)

models_dir = os.path.dirname(__file__)
for file in os.listdir(models_dir):
    path = os.path.join(models_dir, file)
    if not file.startswith('_') and not file.startswith('.') and file.endswith('.py'):
        model_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module('neurst.metrics.' + model_name)
