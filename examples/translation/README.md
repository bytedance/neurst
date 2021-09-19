# Neural Machine Translation

This README contains instructions for preparing parallel data and training neural translation models. 

We take WMT14 EN->DE as an example.


### Contents
* [Requirements](#requirements)
* [Training a transformer model](#training-a-transformer-model)
    * [Download and pre-process data](#download-and-pre-process-data)
    * [Training and validating a transformer model](#training-and-validating-a-transformer-base-model)
    * [Accelerating training with TensorFlow XLA](#accelerating-training-with-tensorflow-xla)
    * [Evaluation on testset](#evaluation-on-testset) 
* [Others](#others)
    * [Word piece](#word-piece)
    * [Compound Split BLEU](#compound-split-bleu)
    * [WMT14 EN2DE Benchmark](#wmt14-en2de-benchmark)
        * [Benchmark models](#benchmark-models)
        * (Base) Tokenized BLEU
        * (Base) Detokenized BLEU (sacreBLEU)
        * (Big) Tokenized BLEU
        * (Big) Detokenized BLEU (sacreBLEU)


## Requirements

**pip**
- TensorFlow >=2.3.0
- subword-nmt
- pyyaml
- sacrebleu
- sacremoses

**others**
```bash
$ git clone https://github.com/moses-smt/mosesdecoder.git
```

## Training a transformer model

### Download and pre-process data
By runing with 
```bash
$ ./examples/translation/prepare-wmt14en2de-bpe.sh /path_to_mosesdecoder
```
we will get the preprocessed training data and raw testsets under directory `wmt14_en_de/`, i.e.
```bash
/wmt14_en_de/
├── codes.bpe  # BPE codes
├── newstest2013.de.txt   # newstest2013 as devset
├── newstest2013.en.txt
├── newstest2014.de.txt  # newstest2014 as testset
├── newstest2014.en.txt
├── prediction_args.yml   # the arguments for prediction
├── train.de.tok.bpe.txtde  # the pre-processed training data
├── train.en.tok.bpe.txt
├── train.de.txt  # the raw training data
├── train.en.txt
├── training_args.yml  # the arguments for training
├── translation_bpe.yml  # the arguments for training data and data pre-processing logic
├── validation_args.yml  # the arguments for validation
├── vocab.de  # the vocabulary
└── vocab.en
```

Here we apply moses tokenizer to the sentences and jointly learn subword units (BPE) with 40K merge operations. 

### Training and validating a transformer-base model
We can directly use the yaml-style configuration files generated above to train and evaluate a transformer model.

```bash
python3 -m neurst.cli.run_exp \
    --config_paths wmt14_en_de/training_args.yml,wmt14_en_de/translation_bpe.yml,wmt14_en_de/validation_args.yml \
    --hparams_set transformer_base \
    --model_dir /wmt14_en_de/benchmark_base
```
where `/wmt14_en_de/benchmark_base` is the root path for checkpoints. Here we use `--hparams_set transformer_base` to train a transformer model including 6 encoder layers and 6 decoder layers with `dmodel=512`.

Alternatively, 
- we can set `--hparams_set transformer_big` to use the `dmodel=1024` version, which usually achives better performance.
- the `transformer_base/big` defines the "pre-norm" transformer structure by default, and we can additionally plus `--encoder.post_normalize` and `--decoder.post_normalize` options to change to the "post-norm" version.

We train the transformer model on multiple GPUs, as long as there is no GPU out-of-memory exception. Moreover, we can set `--update_cycle n --batch_size 32768//n` to simulate `n` GPUs with 1 GPU.

### Accelerating training with TensorFlow XLA

To accelerate the training speed, we can simply enable TensorFlow XLA via `--enable_xla` option and separate the validation procedure from the training, that is
```bash
python3 -m neurst.cli.run_exp \
    --config_paths wmt14_en_de/training_args.yml,wmt14_en_de/translation_bpe.yml \
    --hparams_set transformer_base \
    --model_dir /wmt14_en_de/benchmark_base \
    --enable_xla
```

Then, we start another process with one GPU for validation by
```bash
python3 -m neurst.cli.run_exp \
    --entry validation \
    --config_paths wmt14_en_de/validation_args.yml \
    --model_dir /wmt14_en_de/benchmark_base
```

This process will constantly scan the `model_dir`, evaluate each checkpoint and store the checkpoints with best metrics (i.e. BLEU scores) into `{model_dir}/best` directory along with the corresponding averaged version (by default 10 latest checkpoints) into `{model_dir}/best_avg`. 

### Evaluation on testset
By running with
```bash
python3 -m neurst.cli.run_exp \
    --config_paths wmt14_en_de/prediction_args.yml \
    --model_dir wmt14_en_de/benchmark_base/best_avg
```
BLEU scores will be reported on both dev (newstest2013) and test (newstest2014) set.

## Others

### Word piece
We additionally provide [prepare-wmt14en2de-wp.sh](/examples/translation/prepare-wmt14en2de-wp.sh) to pre-process the data with word piece, which sometimes, achieves better performance.

### Compound Split BLEU
Some research works report their TransformerBig baseline on en->de newstest2014 as 29+. We found that they may repeatedly split compound words during evaluation.

That is, when evaluating with the Compound Split BLEU (like [compound_split_bleu.sh](#https://github.com/pytorch/fairseq/blob/master/scripts/compound_split_bleu.sh)), one may already apply the [moses tokenizer](#https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl) on both hypotheses and references with the `-a` option. The `-a` option enables splitting the compound words.
> After that, "12-year-old" becomes "12 @-@ year @-@ old". 

Then, the `compound_split_bleu.sh` would split the compound words again.
> "12 @-@ year @-@ old" becomes "12 @ ##AT##-##AT## @ year @ ##AT##-##AT## @ old".

It increases the matched n-grams and results in a much higher BLEU score.

Here, we also provide such operation by overwriting the metric option when evaluation (`--metric compound_split_bleu`). But we still recommend to use tokenized BLEU or sacreBLEU.

### WMT14 EN2DE Benchmark

#### Benchmark Models
|                        | hparams | norm type |  |
|------------------------|---------|-----------|----- |
| BPE                    | base    | pre-norm  | \[[LINK](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/translation/wmt14_ende/transformer_base_bpe_prenorm.tgz)\]|
| BPE                    | base    | post-norm | \[[LINK](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/translation/wmt14_ende/transformer_base_bpe_postnorm.tgz)\]|
| BPE                    | big     | pre-norm  | \[[LINK](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/translation/wmt14_ende/transformer_big_bpe_prenorm.tgz)\]|
| BPE                    | big     | post-norm | \[[LINK](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/translation/wmt14_ende/transformer_big_bpe_postnorm.tgz)\]|
| word piece             | base    | pre-norm  | \[[LINK](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/translation/wmt14_ende/transformer_base_wp_prenorm.tgz)\]|
| word piece             | base    | post-norm | \[[LINK](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/translation/wmt14_ende/transformer_base_wp_postnorm.tgz)\]|
| word piece             | big     | pre-norm  | \[[LINK](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/translation/wmt14_ende/transformer_big_wp_prenorm.tgz)\]|
| word piece             | big     | post-norm | \[[LINK](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/translation/wmt14_ende/transformer_big_wp_postnorm.tgz)\]|


#### (Base) Tokenized BLEU
|                        | hparams | norm type | dev(newstest2013) | test(newstest2014) |
|------------------------|---------|-----------|-------------------|--------------------|
| (Vaswani et al., 2017) | base    | post-norm | -                 | 27.3               |
| BPE                    | base    | pre-norm  | 26.2              | 26.8               |
| BPE                    | base    | post-norm | 26.9              | 27.9               |
| word piece             | base    | pre-norm  | 26.2              | 27.0               |
| word piece             | base    | post-norm | 26.6              | 28.0               |

#### (Base) Detokenized BLEU (sacreBLEU)
|                        | hparams | norm type | dev(newstest2013) | test(newstest2014) |
|------------------------|---------|-----------|-------------------|--------------------|
| (Vaswani et al., 2017) | base    | post-norm | -                 | -                  |
| BPE                    | base    | pre-norm  | 25.9              | 26.2               |
| BPE                    | base    | post-norm | 26.6              | 27.3               |
| word piece             | base    | pre-norm  | 26.0              | 26.4               |
| word piece             | base    | post-norm | 26.4              | 27.4               |

#### (Big) Tokenized BLEU
|                        | hparams | norm type | dev(newstest2013) | test(newstest2014) |
|------------------------|---------|-----------|-------------------|--------------------|
| (Vaswani et al., 2017) | big     | post-norm | -                 | 28.4               |
| BPE                    | big     | pre-norm  | 26.7              | 27.7               |
| BPE                    | big     | post-norm | 27.0              | 28.0               |
| word piece             | big     | pre-norm  | 26.6              | 28.2               |
| word piece             | big     | post-norm | 27.0              | 28.3               |

#### (Big) Detokenized BLEU (sacreBLEU)
|                        | hparams | norm type | dev(newstest2013) | test(newstest2014) |
|------------------------|---------|-----------|-------------------|--------------------|
| (Vaswani et al., 2017) | big     | post-norm | -                 | -                  |
| BPE                    | big     | pre-norm  | 26.4              | 27.1               |
| BPE                    | big     | post-norm | 26.8              | 27.4               |
| word piece             | big     | pre-norm  | 26.4              | 27.5               |
| word piece             | big     | post-norm | 26.8              | 27.7               |
