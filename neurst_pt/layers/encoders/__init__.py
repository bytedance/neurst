import importlib
import os

from neurst.utils.registry import setup_registry
from neurst_pt.layers.encoders.encoder import Encoder

build_encoder, register_encoder = setup_registry(Encoder.REGISTRY_NAME, base_class=Encoder, backend="pt")

models_dir = os.path.dirname(__file__)
for file in os.listdir(models_dir):
    path = os.path.join(models_dir, file)
    if not file.startswith('_') and not file.startswith('.') and file.endswith('.py'):
        model_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module('neurst_pt.layers.encoders.' + model_name)
