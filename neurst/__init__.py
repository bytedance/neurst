# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals

import importlib

from .__version__ import __version__  # NOQA

__author__ = "ZhaoChengqi <zhaochengqi.d@bytedance.com>"

__all__ = [
    "cli",
    "data",
    "criterions",
    "exps",
    "layers",
    "metrics",
    "models",
    "tasks",
    "utils",
    "training",
    "optimizers",
    "sparsity"
]

importlib.import_module("neurst.criterions")
importlib.import_module("neurst.data")
importlib.import_module("neurst.data.audio")
importlib.import_module("neurst.data.data_pipelines")
importlib.import_module("neurst.data.datasets")
importlib.import_module("neurst.data.datasets.audio")
importlib.import_module("neurst.data.text")
importlib.import_module("neurst.exps")
importlib.import_module("neurst.layers")
importlib.import_module("neurst.layers.attentions")
importlib.import_module("neurst.layers.decoders")
importlib.import_module("neurst.layers.encoders")
importlib.import_module("neurst.layers.metric_layers")
importlib.import_module("neurst.layers.quantization")
importlib.import_module("neurst.layers.search")
importlib.import_module("neurst.metrics")
importlib.import_module("neurst.models")
importlib.import_module("neurst.optimizers")
importlib.import_module("neurst.optimizers.schedules")
importlib.import_module("neurst.sparsity")
importlib.import_module("neurst.tasks")
importlib.import_module("neurst.training")
importlib.import_module("neurst.utils")
importlib.import_module("neurst.utils.converters")
