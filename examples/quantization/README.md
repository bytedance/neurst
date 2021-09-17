# Machine Translation with Quantization Aware Training

This README contains instructions for fine-tuning and predicting with quantization. Any questions, feel free to contact [weiyang.god@bytedance.com](mailto:weiyang.god@bytedance.com), [wangxiaohui.neo@bytedance.com](mailto:wangxiaohui.neo@bytedance.com).

Note that, in order to realize fully quantized inference, [LightSeq](https://github.com/bytedance/lightseq) is what you want!

Let's take the transformer model as an example.

### Contents
* [Fine-tuning with quantization](#fine-tuning-with-quantization)
* [Evaluation on quantized models](#evaluation-on-quantized-models)
* [View the quantized weights](#view-the-quantized-weights)
* [Implement quantization in your code](#implement-quantization-in-your-code)
    * [Inherit the `QuantLayer` class](#inherit-the-quantlayer-class)
    * [Add the quantizers](#add-the-quantizers)
    * [Get the quantized values](#get-the-quantized-values)


## Fine-tuning with quantization
Assume we have followed the [translation recipe]((#/examples/translation/README.md)) and trained a strong transformer big model at directory `big_wp_prenorm/`

Then, we fine-tune the model by enabling quantization:
```bash
python3 -m neurst.cli.run_exp \
    --config_paths wmt14_en_de/training_args.yml,wmt14_en_de/translation_wordpiece.yml,wmt14_en_de/validation_args.yml \
    --hparams_set transformer_big
    --pretrain_model big_wp_prenorm/ \
    --model_dir big_wp_prenorm_qat/ \
    --initial_global_step 250000 \
    --train_steps 50000 \
    --summary_steps 200 \
    --save_checkpoints_steps 1000 \
    --enable_quant \
    --quant_params "{'quant_strategy':'min/max','quant_bits':8,'quant_weight_clip_max':1.0,'quant_act_clip_max':16.0}"
```
where `pretrain_model` is the directory of pre-trained machine translation checkpoint. We set `initial_global_step` to the final step of the `pretrain_model` for smaller learning rate when fine-tuning. We can enable the quantization via `--enable_quant` and set the quantization parameters via `--quant_params`.

Currently supported quantization parameters are `quant_strategy`, `quant_bits`, `quant_weight_clip_max` and `quant_act_clip_max`, where `quant_strategy` only supports `min/max` now. Empirically, the `quant_weight_clip_max` and `quant_act_clip_max` are usually set to 1.0 and 16.0 for Transformer BIG respectively.

Note that the quantization may slow down the fine-tuning, but it will not take too much time to reach convergence.

## Evaluation on quantized models
By running with
```bash
python3 -m neurst.cli.run_exp \
    --config_paths wmt14_en_de/prediction_args.yml \
    --model_dir big_wp_prenorm_qat/best
```

Evaluation on quantized models is the same as normal models, where the parameters for quantization are already stored in `big_wp_prenorm_qat/best/model_configs.yml`. Note that it will not accelerate the inference but is only used to verify the validity of quantization. In order to realize fully quantized inference, [LightSeq](https://github.com/bytedance/lightseq) is the good choice.


## View the quantized weight
Assume we have a quantized checkpoint or download [transformer_big_wp_prenorm_int8quant](#http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/translation/wmt14_ende/transformer_big_wp_prenorm_int8quant.tgz). 
Pass the model directory to `examples/quantization/example_view_quant_weight.py`, the quantized weight will be displayed:
```bash
python3 examples/quantization/example_view_quant_weight.py big_wp_prenorm_int8quant/
```
Then, we get
```
The quantized weight of encoder layer0's first feedforward
QuantizeV2(output=<tf.Tensor: shape=(1024, 4096), dtype=qint8, numpy=
array([[ -5,  -5,   0, ...,   8,  -5,   3],
       [  0,  -3, -10, ...,   5,  -8,   0],
       [ 10,  -8, -10, ...,  -8, -10,   0],
       ...,
       [  3,  -5,   0, ...,  -3,  -5,  -8],
       [ -3,  -3,  -3, ...,  -8,  -5, -10],
       [  0,   3,   3, ...,   8,  -5,   5]], dtype=int8)>, 
   output_min=<tf.Tensor: shape=(), dtype=float32, numpy=-0.39481023>, 
   output_max=<tf.Tensor: shape=(), dtype=float32, numpy=0.39172578>)
```


## Implement quantization in your code
Taking `MultiHeadDenseLayer` as an example, the code is showed as following:
```python
from neurst.layers.quantization.quant_layers import QuantLayer


class MultiHeadDenseLayer(QuantLayer):

    def build(self, input_shape):
        # Define the weights without any modification.
        self._kernel = self.add_weight(
            "kernel",
            shape=shape,
            activation=activation,
            initializer=initializer,
            trainable=True)
        # Define the quantizers for activations.
        self.add_activation_quantizer(name="output", activation=activation)
        super(MultiHeadDenseLayer, self).build(input_shape)

    def call(self, inputs):
        # Quantize the weights before using them.
        kernel = self.quant_weight(self._kernel)
        outputs = func(kernel, inputs)
        # Quantize the activations after calculating them.
        outputs = self.quant(outputs, name="output")
        return outputs
```

In summary, quantization is divided into three steps.

### Inherit the `QuantLayer` class
Firstly, one should inherit the `QuantLayer` class if it originally inherited the `tf.keras.layer.Layer` class.

### Add the quantizers
In `build` function, any modification on `self.add_weight` is not required. In the quantization implementation, every weight except `bias` will be created together with a quantizer having the same name. 

One need to manually add an activation quantizer if needed using `self.add_activation_quantizer`. The function has two parameters, in which `name` must be different between each other and `activation` is used to specify which type of quantizer to use. Currently we support four types, `act`, `relu`, `softmax` (or None for `act` by default), which used for normal activations, relu and softmax respectively. 

### Get the quantized values
Just use `self.quant(inputs, name)` and `self.quant_weight(weight)` (for variables created by `self.add_weight`) to get the quantized values. Note that the `name` must be the same as defined in `build` function.

If keras layers are used, one must manually implement it to quantize the weights and activations inside it. An example of quantized `tf.keras.layer.Dense` can be found here [quant_dense_layer](/neurst/layers/quantization/quant_dense_layer.py).
