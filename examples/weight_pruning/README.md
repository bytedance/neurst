# Model Optimization - Weight Pruning

The weight pruning API in NeurST is built as an optimizer wrapper, which is already integrated in the default trainer.

> For more details about pruning technique: https://blog.tensorflow.org/2019/05/tf-model-optimization-toolkit-pruning-API.html

Let's take the transformer model as an example. 

### Pruning Configurations

Assume we have followed the [translation recipe](/examples/translation/README.md) and trained a strong transformer big model at directory `big_wp_prenorm/`

Then, we fine-tune the model by enabling weight pruning:
``` bash
python3 -m neurst.cli.run_exp \
    --config_paths wmt14_en_de/training_args.yml,wmt14_en_de/translation_wordpiece.yml,wmt14_en_de/validation_args.yml \
    --hparams_set transformer_big
    --pretrain_model big_wp_prenorm/ \
    --model_dir big_wp_prenorm_prune_1/ \
    --initial_global_step 250000 \
    --train_steps 10000 \
    --summary_steps 200 \
    --save_checkpoints_steps 400 \
    --pruning_schedule polynomial_decay \
    --initial_sparsity 0 \
    --target_sparsity 0.1 \
    --begin_pruning_step 0 \
    --end_pruning_step 500 \
    --pruning_frequency 100 \
    --nopruning_variable_pattern (ln/gamma)|(ln/beta)|(modalit)
```
	
Here we follow the command for training and overwrite several options:

- `pretrain_model`: restore the parameters from a well-trained model;
- `initial_global_step`: start from non-zero step number, which controls the learning rate according to the Noam schedule;
- `train_step`: we only fine-tune a small number of steps;
- `pruning_schedule`: the pruning schedule with a PolynomialDecay function;
- `initial_sparsity`: the sparsity at which pruning begins;
- `target_sparsity`: the sparsity at which pruning ends;
- `begin_pruning_step`: step at which to begin pruning (start from 0 not the `initial_global_step`);
- `end_pruning_step`: step at which to end pruning (start from 0 not the `initial_global_step`);
- `pruning_frequency`: only update the pruning mask every this steps;
- `nopruning_variable_pattern`: a regular expression that indicates the variables will be pruned. Alternatively, we can use `--pruning_variable_pattern` to select the variables will be pruned. Note that `nopruning_variable_pattern` will take effect only if `pruning_variable_pattern` is not provided (default `None`).

Same as the translation recipe, the averaged checkpoint with best BLEU on devset stores in `big_wp_prenorm_prune_1/best_avg/`.

### Performance

We use the pre-norm transformer-big model based on word piece ([LINK](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/translation/wmt14_ende/transformer_big_wp_prenorm.tgz)) from [translation/README.md](/examples/translation/README.md) and test the sparsity from 0.1~0.5.

The performance is listed below:

**Tokenized BLEU**

| sparsity | dev(newstest2013) | test(newstest2014) |
|----------|-------------------|--------------------|
|   [0.0](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/translation/wmt14_ende/transformer_big_wp_prenorm.tgz)    | 26.6              | 28.2               |
|   [0.1](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/translation/wmt14_ende/transformer_big_wp_prenorm_prune1.tgz)    | 26.6              | 28.1               |
|   [0.2](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/translation/wmt14_ende/transformer_big_wp_prenorm_prune2.tgz)    | 26.6              | 28.2               |
|   [0.3](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/translation/wmt14_ende/transformer_big_wp_prenorm_prune3.tgz)    | 26.6              | 28.0               |
|   [0.4](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/translation/wmt14_ende/transformer_big_wp_prenorm_prune4.tgz)    | 26.6              | 27.9               |
|   [0.5](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/translation/wmt14_ende/transformer_big_wp_prenorm_prune5.tgz)    | 26.5              | 27.7               |


**Deokenized BLEU (sacreBLEU)**

| sparsity | dev(newstest2013) | test(newstest2014) |
|----------|-------------------|--------------------|
|   [0.0](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/translation/wmt14_ende/transformer_big_wp_prenorm.tgz)    | 26.4              | 27.5               |
|   [0.1](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/translation/wmt14_ende/transformer_big_wp_prenorm_prune1.tgz)    | 26.4              | 27.4               |
|   [0.2](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/translation/wmt14_ende/transformer_big_wp_prenorm_prune2.tgz)    | 26.4              | 27.5               |
|   [0.3](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/translation/wmt14_ende/transformer_big_wp_prenorm_prune3.tgz)    | 26.4              | 27.3               |
|   [0.4](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/translation/wmt14_ende/transformer_big_wp_prenorm_prune4.tgz)    | 26.4              | 27.3               |
|   [0.5](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/translation/wmt14_ende/transformer_big_wp_prenorm_prune5.tgz)    | 26.3              | 27.1               |
