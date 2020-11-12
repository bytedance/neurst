import importlib
import os

from neurst.criterions.criterion import Criterion
from neurst.utils.registry import setup_registry

build_criterion, register_criterion = setup_registry(Criterion.REGISTRY_NAME, base_class=Criterion,
                                                     verbose_creation=True)

models_dir = os.path.dirname(__file__)
for file in os.listdir(models_dir):
    path = os.path.join(models_dir, file)
    if not file.startswith('_') and not file.startswith('.') and file.endswith('.py'):
        model_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module('neurst.criterions.' + model_name)
