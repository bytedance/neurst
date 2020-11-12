from collections import namedtuple

METRIC_REDUCTION = namedtuple(
    "metric_reduction", "SUM MEAN")(0, 1)

REGISTERED_METRICS = dict()


def register_metric(name, redution):
    if name in REGISTERED_METRICS:
        raise ValueError(f"Metric {name} already registered.")
    REGISTERED_METRICS[name] = redution


def get_metric_reduction(name, default=METRIC_REDUCTION.MEAN):
    return REGISTERED_METRICS.get(name, default)
