import importlib
import os

from neurst.utils.registry import setup_registry

_, register_agent = setup_registry("simuleval_agent", verbose_creation=False)

models_dir = os.path.dirname(__file__)
for file in os.listdir(models_dir):
    path = os.path.join(models_dir, file)
    if not file.startswith('_') and not file.startswith('.') and file.endswith('.py'):
        model_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module('neurst.utils.simuleval_agents.' + model_name)
