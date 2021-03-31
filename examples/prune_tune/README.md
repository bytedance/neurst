# Prune-Tune
Official code repository for AAAI2021 paper:

[Finding Sparse Structures for Domain Specific Neural Machine Translation](https://arxiv.org/abs/2012.10586)

[Here](https://ohlionel.github.io/project/Prune-Tune/) is a brief introduction of Prune-Tune.

This project is based on [Neurst](https://github.com/bytedance/neurst), an open source Neural Speech Translation Toolkit. 

Here is an example to train a general model for En-De Translation, then adapt to a target domain(novel) via Prune-Tune.

## Neurst Installation
Install from source:
```bash
git clone https://github.com/bytedance/neurst.git
cd neurst
pip3 install -r requirements.txt
```
please see installation details in [Neurst](https://github.com/bytedance/neurst)

## Data Preprocess
Datasets:
|   Domain  |  Dataset | Download|
|  ----  | ----  | :----:|
| General Domain   | WMT14(En-De) | Automatic |
| Target Domain  | Novel/EMEA/IWSLT14 | [Link](https://github.com/ohlionel/Prune-Tune/tree/main/neurst/data)


<!-- 
General Domain: WMT14(En-De)

Target Domain: [Novel Dataset](https://opus.nlpl.eu/Books.php) from OPUS -->

By runing with
```bash
# Download wmt14(en2de) dataset, learn wordpiece vocabulary, and preprocess data.
bash ./examples/prune_tune/scripts/prepare-wmt14en2de-wp.sh 

# Download novel.tar.gz to 'data/novel.tar.gz' via link above.

# Unzip novel dataset 
tar -zxvf data/novel.tar.gz -C data/ 

# Use the wordpiece vocabulary learned above.
cp data/wmt14_en_de/vocab data/novel/ 

# Preprocess novel data.
bash ./examples/prune_tune/scripts/prepare-target-dataset-wp.sh novel
```

we will get the preprocessed training data and raw testsets under directory `data/wmt14_en_de/` and `data/novel`: 
```bash
data/wmt14_en_de/
├── vocab  # wordpiece codes
├── newstest2013.de.txt   # newstest2013 as devset
├── newstest2013.en.txt
├── newstest2014.de.txt  # newstest2014 as testset
├── newstest2014.en.txt
├── prediction_args.yml   # the arguments for prediction
├── train.de.txt  # the raw training data
├── train.en.txt
├── training_args.yml  # the arguments for training
├── translation_wordpiece.yml  # the arguments for training data and data pre-processing logic
├── validation_args.yml  # the arguments for validation
├── training_records # directory of training TFRecords
    ├──train.tfrecords.00000
    ├──train.tfrecords.00001
    ├── ...
├── ...
```
Note: It may take a few hours to complete data preprocess.

## Train the General Domain Model
We can directly use the yaml-style configuration files generated above to train a general domain model on WMT14(En-De).
```bash
cd neurst
python3 -m neurst.cli.run_exp \
    --include examples/prune_tune/src/ \
    --entry prune_tune_train \
    --config_paths data/wmt14_en_de/training_args.yml,data/wmt14_en_de/translation_wordpiece.yml,data/wmt14_en_de/validation_args.yml \
    --hparams_set transformer_big \
    --model_dir models/benchmark_big
```
You may use `CUDA_VISIBLE_DEVICES` flag to run on sepecific gpu devices.
## Prune the General Domain Model 
We can simply prune a model with Neurst, see [Weight Pruning](https://github.com/ohlionel/Prune-Tune/tree/main/neurst/examples/weight_pruning) for details.
```bash
python3 -m neurst.cli.run_exp \
    --include examples/prune_tune/src/ \
    --entry prune_tune_train \
    --config_paths data/wmt14_en_de/training_args.yml,data/wmt14_en_de/translation_wordpiece.yml,data/wmt14_en_de/validation_args.yml \
    --hparams_set transformer_big \
    --pretrain_model models/benchmark_big/best/ \
    --model_dir models/sparsity_10 \
    --checkpoints_max_to_keep 3 \
    --initial_global_step 250000 \
    --train_steps 10000 \
    --summary_steps 200 \
    --save_checkpoints_steps 500 \
    --pruning_schedule polynomial_decay \
    --initial_sparsity 0 \
    --target_sparsity 0.1 \
    --begin_pruning_step 0 \
    --end_pruning_step 5000 \
    --pruning_frequency 100 \
    --nopruning_variable_pattern "(ln/gamma)|(ln/beta)|(modalit)"  # No pruning to LayerNorm/Embedding Layers
```
We will get the pruned model `models/sparsity_10` in which 10% of parameters is pruned. `sparsity_10/mask.pkl` save all binary pruning masks, where 0 indicates zero-value pruned weight.

## Partially Tune the Model with Taget Domian Dataset
According to the pruning mask file `sparsity_10/mask.pkl`, we can only update those pruned weight during tuning. 
```bash
python3 -m neurst.cli.run_exp \
    --include examples/prune_tune/src/ \
    --entry prune_tune_train \
    --config_paths data/novel/training_args.yml,data/novel/translation_wordpiece.yml,data/novel/validation_args.yml \
    --hparams_set transformer_big \
    --pretrain_model models/sparsity_10 \
    --model_dir models/sparsity_10_novel \
    --initial_global_step 0 \
    --train_steps 10000 \
    --summary_steps 200 \
    --save_checkpoints_steps 1000 \
    --partial_tuning \
    --mask_dir models/sparsity_10/mask.pkl 
```
## Evalution on General and Target Domain
Evaluate on target domain with full model:
```bash
python3 -m neurst.cli.run_exp \
    --include examples/prune_tune/src/ \
    --entry mask_predict \
    --config_paths data/novel/prediction_args.yml \
    --model_dir models/sparsity_10_novel/best
```
Evaluate on general domain with the general sub-network:
```bash
python3 -m neurst.cli.run_exp \
    --include examples/prune_tune/src/ \
    --entry mask_predict \
    --config_paths data/wmt14_en_de/prediction_args.yml \
    --model_dir models/sparsity_10_novel/best \
    --mask_dir models/sparsity_10/mask.pkl \
    --apply_mask
```

## Checkpoints Download

|#|Model| Pre-trained Model | Dataset | Approach | Train Steps | WMT_BLEU | TEST_BLEU | Download Links|
|----|----|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|1|benchmark_big	|-|	wmt14|	training from scratch|	500000	|28.46|	-| [benchmark_big.tar.gz](https://drive.google.com/file/d/1a8YWn9fMT7VEV1Oh-S9WS9Z6kxeVdIBw/view?usp=sharing) |	
2 | sparsity_10	|#1|	wmt14|	gradual pruning|	10000|	28.49|	-| [sparsity_10.tar.gz](https://drive.google.com/file/d/1EGVACputsPdoiPcgw5oiWWKkMOn_2WCo/view?usp=sharing) |	
3	|sparsity_10_iwslt|	#2|	iwslt14	|partial tuning	|10000|	28.49|	31.34	| [sparsity_10_iwslt.tar.gz](https://drive.google.com/file/d/1SthoHiv7p_O926TeBzDArn_ZbwDgsWru/view?usp=sharing) |
4|	sparsity_10_emea	|#2	|emea	|partial tuning	|10000|	28.49	|30.85| [sparsity_10_emea.tar.gz](https://drive.google.com/file/d/13fbgJLgJkHoK1CKEmbPDi6L6x3GPGkRU/view?usp=sharing) |
5	|sparsity_10_novel|	#2	|novel|	partial tuning	|10000|	28.49	|24.19|[sparsity_10_novel.tar.gz](https://drive.google.com/file/d/17y6vxBr7tVXqDjahlKOwdTBTA1AjmMPW/view?usp=sharing) |

## Citation
Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follows.
```
@inproceedings{jianze2021prunetune,
  title={Finding Sparse Structures for Domain Specific Neural Machine Translation},
  author={Jianze Liang, Chengqi Zhao, Mingxuan Wang, Xipeng Qiu, Lei Li},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2021}
}
```





