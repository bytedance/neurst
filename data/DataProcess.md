

# Data Prepare for Multilingual Machine Translation

 
### Requirements

**pip**

- SentencePiece

  

### Step 1: Download Data

  

First, we shall download raw parallel bilingual text data into the dataset folder. Here we give bash scripts of downloading OPUS-100 and IWSLT, for example.

  

Then, we get the parallel txt file via `get_parallel.py`. This script will concatenate the language token (*e.g.* `<en>`) to the beginning of the source sentence. 
 
 *In case of OPUS-100:*
 ```bash
python3 get_parallel.py --input_path=/dataset/opus-100 --output_path=opus_train --mode=train
python3 get_parallel.py --input_path=/dataset/opus-100 --output_path=opus_dev --mode=dev
python3 get_parallel.py --input_path=/dataset/opus-100 --output_path=opus_test --mode=test
```

### Step 2: Clean Data

  

Note we shall clean and normalize the downloaded data with Moses, where some download scripts have already offered the cleaned data.

Alternatively, you can use `clean_data.py --moses_path=<MOSES_DIR> --input_path=<INPUT_DIR>` to automatically get the bash scripts for cleaning data. 

*In case of OPUS-100:*
```bash 
python3 clean_data.py --moses_path=./mosesdecoder/ --input_path=opus_train
python3 clean_data.py --moses_path=./mosesdecoder/ --input_path=opus_dev
python3 clean_data.py --moses_path=./mosesdecoder/ --input_path=opus_test --suffix=in
# copy the raw reference to the target folder after running the clean commands printed from the previous sripts
# cp opus_test/*.out opus_test_clean
```


*Caution*:  The reference(target) of the test set shall not be cleaned. You shall run `clean_data.py --moses_path=<MOSES_DIR> --input_path=<INPUT_DIR> --suffix=in` to process the input source of test set only. 

*Caution*: [Fairseq](https://github.com/pytorch/fairseq/blob/master/scripts/compound_split_bleu.sh) and [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/get_ende_bleu.sh) used compound spitting trick for historical reasons, where the trick was applied to the test sets.  

### Step 3: BPE encoding with SentencePiece

You can concatenate all training files into a single file, or you can downsample according to the size of each language:

#### (Optional) Downsample the Corpus

We can downsample the corpus of each language via the policy of [1]
The downsample is only operated on the training set.


BPE encoding the sampled/concanated file
```bash
python3 downsample.py --input_path=opus_train_clean/
```
the output file is `samples`

#### BPE with SentencePiece

Get BPE code file with SentencePiece via `sentencePiece_train.py`. 

for OPUS-100
```bash
python3 sentencePiece_train.py --input_file=samples --output_path=spm_bpe --vocab_size=32000
```
The default vocab size is 32000

#### (Optional for WMT) Remove Mismatched Sentences
Some data in WMT are mismatched. We can tell it by the length ratio of the source and target. We can remove such mismatched bilingual parallel sentences.


### Reference
[1] Arivazhagan, Naveen, Ankur Bapna, Orhan Firat, Dmitry Lepikhin, Melvin Johnson, Maxim Krikun, Mia Xu Chen et al. "Massively multilingual neural machine translation in the wild: Findings and challenges."  _arXiv preprint arXiv:1907.05019_  (2019).