import torch.nn as nn

from neurst.utils.registry import setup_registry
from neurst_pt.layers.attentions.multi_head_attention import MultiHeadAttention, MultiHeadSelfAttention
from neurst_pt.layers.common_layers import PrePostProcessingWrapper, TransformerFFN

build_base_layer, register_base_layer = setup_registry("base_layer", base_class=nn.Module,
                                                       verbose_creation=False, backend="pt")

register_base_layer(MultiHeadSelfAttention)
register_base_layer(MultiHeadAttention)
register_base_layer(TransformerFFN)


def build_transformer_component(layer_args,
                                norm_shape,
                                dropout_rate,
                                pre_norm=True,
                                epsilon=1e-6):
    base_layer = build_base_layer(layer_args)
    return PrePostProcessingWrapper(
        layer=base_layer,
        norm_shape=norm_shape,
        dropout_rate=dropout_rate,
        epsilon=epsilon,
        pre_norm=pre_norm)
