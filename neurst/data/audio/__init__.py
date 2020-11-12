import importlib
import os

from neurst.data.audio.feature_extractor import FeatureExtractor
from neurst.utils.registry import setup_registry

build_feature_extractor, register_feature_extractor = setup_registry(
    "feature_extractor", base_class=FeatureExtractor, verbose_creation=True)

models_dir = os.path.dirname(__file__)
for file in os.listdir(models_dir):
    path = os.path.join(models_dir, file)
    if not file.startswith('_') and not file.startswith('.') and file.endswith('.py'):
        model_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module('neurst.data.audio.' + model_name)
