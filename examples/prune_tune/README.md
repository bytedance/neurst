# Prune-Tune: Finding Sparse Structures for Domain Specific NMT
This example shows how to run the [Prune-Tune](https://arxiv.org/abs/2012.10586) (Liang et al., 2021) method that first **prunes** the NMT model and then **tunes** partial model parameters to learn domain-specific knowledge. [Here](https://ohlionel.github.io/project/Prune-Tune/) is the brief introduction of the Prune-Tune method.

The Prune-Tune method is from the following paper.
```
@inproceedings{jianze2021prunetune,
  title={Finding Sparse Structures for Domain Specific Neural Machine Translation},
  author={Jianze Liang, Chengqi Zhao, Mingxuan Wang, Xipeng Qiu, Lei Li},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2021}
}
```

## Datasets, results and available models
In this recipe, we will show how to adapt a NMT model of general domain to various target domains using Prune-Tune.

Datasets:
- The general domain model is trained on WMT14 en->de dataset.
- The target domain contains Novel, EMEA and Oral (IWSLT14). The datasets can be downloaded from [HERE](https://github.com/ohlionel/Prune-Tune/tree/main/neurst/data).

We report tokenized BLEU. The baseline model is Transformer big.

|#|Model| Pre-trained Model | Dataset | Approach | Train Steps | newstest2014 BLEU | target domain BLEU | 
|----|----|:----:|:----:|:----:|:----:|:----:|:----:|
|1| [Baseline](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/prune_tune/transformer_big_baseline.tgz)	|-|	wmt14|	training from scratch|	500000	|28.4|	-|	
2 | [Baseline_pruned10](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/prune_tune/transformer_big_baseline_pruned10.tgz)	|#1|	wmt14|	gradual pruning|	10000|	28.5|	-|  |	
3	|[IWSLTspec_tune10](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/prune_tune/transformer_big_baseline_pruned10iwslt.tgz)|	#2|	iwslt14	|partial tuning	|10000|	28.5|	31.4	|
4|	[EMEAspec_tune10](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/prune_tune/transformer_big_baseline_pruned10emea.tgz)	|#2	|emea	|partial tuning	|10000|	28.5	|30.9| 
5	|[NOVELspec_tune10](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/prune_tune/transformer_big_baseline_pruned10novel.tgz)|	#2	|novel|	partial tuning	|10000|	28.5	|24.2|

## Run the Prune-Tune method

### Train the general domain model and prune
Following the [Weight Pruning](/examples/weight_pruning/README.md), assume we have a well-trained transformer big model on WMT14 en->de dataset with 10% parameters pruned [[LINK](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/neurst/prune_tune/transformer_big_baseline_pruned10.tgz)] and the vocabulary is built using word piece.

Note that, we should add three extra options when pruning:
```bash
--include examples/prune_tune/src/ \ 
--entry prune_tune_train \
--nopruning_variable_pattern "(ln/gamma)|(ln/beta)|(modalit)"  # No pruning to LayerNorm/Embedding Layers
```
Then, we will get the pruned model `transformer_big_baseline_pruned10/` in which 10% of parameters are pruned and the weight masks are saved into `transformer_big_baseline_pruned10/mask.pkl`, where 0 indicates zero-value pruned weight. 

### Prepare domain dataset

Download the datasets with specific domains from [HERE](https://github.com/ohlionel/Prune-Tune/tree/main/neurst/data).

```bash
# Download novel.tar.gz via the link above.

# Untar novel dataset
tar -zxvf novel.tar.gz

# Preprocess novel data.
bash ./examples/prune_tune/scripts/prepare-target-dataset-wp.sh novel/
```

we will get the preprocessed training data and raw testsets under directory `novel/`: 
```bash
data/wmt14_en_de/
├── dev.de
├── dev.en
├── prediction_args.yml   # the arguments for prediction
├── test.de  # the raw training data
├── test.en
├── train.de  # the raw training data
├── train.en
├── training_args.yml  # the arguments for training
├── training_records # directory of training TFRecords
    ├──train.tfrecords-00000-of-00032
    ├──train.tfrecords.00001-of-00032
    ├── ...
├── translation_wordpiece.yml  # the arguments for training data and data pre-processing logic
└── validation_args.yml  # the arguments for validation
```

### Partially tune the model with target domian dataset
According to the mask file `transformer_big_baseline_pruned10/mask.pkl`, we can tune the model parameters only at the masked positions.
```bash
python3 -m neurst.cli.run_exp \
    --include examples/prune_tune/src/ \
    --entry prune_tune_train \
    --config_paths novel/training_args.yml,novel/translation_wordpiece.yml,novel/validation_args.yml \
    --hparams_set transformer_big \
    --pretrain_model transformer_big_baseline_pruned10/ \
    --model_dir transformer_big_baseline_pruned10_novel/ \
    --initial_global_step 0 \
    --train_steps 10000 \
    --summary_steps 200 \
    --save_checkpoints_steps 1000 \
    --partial_tuning \
    --mask_pkl transformer_big_baseline_pruned10/mask.pkl 
```
### Evaluation on the general and target domain
- To evaluate on target domain with full model:
```bash
python3 -m neurst.cli.run_exp \
    --include examples/prune_tune/src/ \
    --entry mask_predict \
    --config_paths novel/prediction_args.yml \
    --model_dir transformer_big_baseline_pruned10_novel/best
```
or
```bash
python3 -m neurst.cli.run_exp \
    --entry predict \
    --config_paths novel/prediction_args.yml \
    --model_dir transformer_big_baseline_pruned10_novel/best
```

- To evaluate on general domain with the general sub-network:
```bash
python3 -m neurst.cli.run_exp \
    --include examples/prune_tune/src/ \
    --entry mask_predict \
    --config_paths wmt14_en_de/prediction_args.yml \
    --model_dir transformer_big_baseline_pruned10_novel/best \
    --mask_pkl transformer_big_baseline_pruned10_novel/mask.pkl
```

