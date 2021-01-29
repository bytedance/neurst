# Copyright 2020 ByteDance Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#!/usr/bin/env bash
set -e

THIS_DIR="$( cd "$( dirname "$0" )" && pwd )"

TOKENIZER=$1/scripts/tokenizer/tokenizer.perl
if [[ ! -f ${TOKENIZER} ]]; then
    echo "Fail to find Moses tokenizer: ${TOKENIZER}"
    echo "Usage: ./prepare-wmt14en2de-bpe.sh MOSES_DIR"
    exit 1
fi

pip3 install -e $THIS_DIR/../../ --no-deps

CLEAN=$1/scripts/training/clean-corpus-n.perl
NORM_PUNC=$1/scripts/tokenizer/normalize-punctuation.perl
REMOVE_NON_PRINT_CHAR=$1/scripts/tokenizer/remove-non-printing-char.perl

DATA_PATH=wmt14_en_de
mkdir -p ${DATA_PATH}
DATA_PATH="$( cd "$DATA_PATH" && pwd )"

# download data and learn word piece vocabulary
python3 $THIS_DIR/download_wmt14en2de.py --output_dir $DATA_PATH

TRAIN_SRC=$DATA_PATH/train.en.txt
TRAIN_TRG=$DATA_PATH/train.de.txt
DEV_SRC=$DATA_PATH/newstest2013.en.txt
DEV_TRG=$DATA_PATH/newstest2013.de.txt
TEST_SRC=$DATA_PATH/newstest2014.en.txt
TEST_TRG=$DATA_PATH/newstest2014.de.txt

echo "shuffling..."
get_seeded_random()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}

random_source=`date +%N`
shuf --random-source=<(get_seeded_random $random_source) \
    $DATA_PATH/train.en.txt > $DATA_PATH/train.en.shuf
shuf --random-source=<(get_seeded_random $random_source) \
    $DATA_PATH/train.de.txt > $DATA_PATH/train.de.shuf

mv $DATA_PATH/train.en.shuf $TRAIN_SRC
mv $DATA_PATH/train.de.shuf $TRAIN_TRG

# pre-process training data
echo "pre-processing train data..."

function tokenize() {
    INP=$1
    LANG=$2
    OUT=$3
    cat $INP | \
        perl $NORM_PUNC $LANG | \
        perl $REMOVE_NON_PRINT_CHAR | \
        perl $TOKENIZER -threads 10 -a -l $LANG -no-escape -q > $OUT
}

$(tokenize $TRAIN_SRC "en" $DATA_PATH/train.en.tok.txt) &
$(tokenize $TRAIN_TRG "de" $DATA_PATH/train.de.tok.txt) &
wait

echo "learn joint 40k BPE..."
subword-nmt learn-joint-bpe-and-vocab \
    --input $DATA_PATH/train.en.tok.txt $DATA_PATH/train.de.tok.txt \
    --symbols 40000 \
    --output $DATA_PATH/codes.bpe \
    --write-vocabulary $DATA_PATH/vocab.en $DATA_PATH/vocab.de

echo "apply BPE..."
subword-nmt apply-bpe --codes $DATA_PATH/codes.bpe \
    --vocabulary $DATA_PATH/vocab.en < $DATA_PATH/train.en.tok.txt > $DATA_PATH/train.en.tok.bpe.txt &
subword-nmt apply-bpe --codes $DATA_PATH/codes.bpe \
    --vocabulary $DATA_PATH/vocab.de < $DATA_PATH/train.de.tok.txt > $DATA_PATH/train.de.tok.bpe.txt &
wait

cp $THIS_DIR/training_args.yml $DATA_PATH/training_args.yml

cat $THIS_DIR/validation_args.yml | \
    sed "s#DEV_SRC#$DATA_PATH/newstest2013.en.txt#" | \
    sed "s#DEV_TRG#$DATA_PATH/newstest2013.de.txt#" > $DATA_PATH/validation_args.yml

cat $THIS_DIR/prediction_args.yml | \
    sed "s#DEV_SRC#$DATA_PATH/newstest2013.en.txt#" | \
    sed "s#DEV_TRG#$DATA_PATH/newstest2013.de.txt#" | \
    sed "s#TEST_SRC#$DATA_PATH/newstest2014.en.txt#" | \
    sed "s#TEST_TRG#$DATA_PATH/newstest2014.de.txt#" > $DATA_PATH/prediction_args.yml

echo "
dataset.class: ParallelTextDataset
dataset.params:
  src_file: $DATA_PATH/train.en.tok.bpe.txt
  trg_file: $DATA_PATH/train.de.tok.bpe.txt
  data_is_processed: True

task.class: translation
task.params:
  batch_by_tokens: True
  batch_size: 32768
  max_src_len: 128
  max_trg_len: 128
  src_data_pipeline.class: TextDataPipeline
  src_data_pipeline.params:
    language: en
    tokenizer: moses
    subtokenizer: bpe
    subtokenizer_codes: $DATA_PATH/codes.bpe
    vocab_path: $DATA_PATH/vocab.en
  trg_data_pipeline.class: TextDataPipeline
  trg_data_pipeline.params:
    language: de
    tokenizer: moses
    subtokenizer: bpe
    subtokenizer_codes: $DATA_PATH/codes.bpe
    vocab_path: $DATA_PATH/vocab.de
" > $DATA_PATH/translation_bpe.yml
