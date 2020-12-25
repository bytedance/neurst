import importlib
import os

from neurst.utils.converters.converter import Converter
from neurst.utils.registry import setup_registry

build_converter, register_converter = setup_registry(Converter.REGISTRY_NAME, base_class=Converter,
                                                     verbose_creation=False, create_fn="new")

models_dir = os.path.dirname(__file__)
for file in os.listdir(models_dir):
    path = os.path.join(models_dir, file)
    if not file.startswith('_') and not file.startswith('.') and file.endswith('.py'):
        model_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module('neurst.utils.converters.' + model_name)
