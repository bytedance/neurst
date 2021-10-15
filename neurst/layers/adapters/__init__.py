import importlib
import os
from neurst.layers.adapters.adapter import Adapter
from neurst.utils.registry import setup_registry

build_adapter, register_adapter = setup_registry("adapter", base_class=Adapter)

models_dir = os.path.dirname(__file__)
for file in os.listdir(models_dir):
    path = os.path.join(models_dir, file)
    if not file.startswith('_') and not file.startswith('.') and file.endswith('.py'):
        model_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module('bytedseq.layers.adapters.' + model_name)
