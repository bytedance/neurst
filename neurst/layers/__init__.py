import tensorflow as tf

from neurst.layers.attentions.light_convolution_layer import LightConvolutionLayer
from neurst.layers.attentions.multi_head_attention import MultiHeadAttention, MultiHeadSelfAttention
from neurst.layers.common_layers import PrePostProcessingWrapper, TransformerFFN
from neurst.utils.registry import setup_registry

build_base_layer, register_base_layer = setup_registry("base_layer", base_class=tf.keras.layers.Layer,
                                                       verbose_creation=False)

register_base_layer(MultiHeadSelfAttention)
register_base_layer(MultiHeadAttention)
register_base_layer(TransformerFFN)
register_base_layer(LightConvolutionLayer)


def build_transformer_component(layer_args,
                                dropout_rate,
                                pre_norm=True,
                                epsilon=1e-6,
                                res_conn_factor=1.,
                                name_postfix=None):
    base_layer = build_base_layer(layer_args)
    return PrePostProcessingWrapper(
        layer=base_layer,
        dropout_rate=dropout_rate,
        epsilon=epsilon,
        pre_norm=pre_norm,
        res_conn_factor=res_conn_factor,
        name=base_layer.name + (name_postfix or "_prepost_wrapper"))
