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

MOSES_TOKENIZER=$1/scripts/tokenizer/tokenizer.perl

if [[ ! -f $MOSES_TOKENIZER ]]; then
    echo "Fail to find Moses tokenizer: ${MOSES_TOKENIZER}"
    echo "Usage: ./03-preprocess.sh MOSES_DIR ROOT_DATA_PATH TRG_LANG"
    exit 1
fi

if [[ ! -n "$2" ]] ;then
    echo "Usage: ./03-preprocess.sh MOSES_DIR ROOT_DATA_PATH TRG_LANG"
    exit 1
else
    DATA_PATH="$2"
fi

if [[ ! -n "$3" ]] ;then
    echo "Usage: ./03-preprocess.sh MOSES_DIR ROOT_DATA_PATH TRG_LANG"
    exit 1;
else
    TRG_LANGUAGE="$3"
fi

RAW_DATA_PATH=$DATA_PATH/raw
TRANSCRIPT_PATH=$DATA_PATH/transcripts/$TRG_LANGUAGE

function makeDirs(){
    if [[ $1 == hdfs://* ]]; then
        hadoop fs -mkdir -p $1
    else
        mkdir -p $1
    fi
}

function fileExists(){
    # DATA_PATH, FILE_NAME
    ABS_PATH=$1/$2
    if [[ -f $ABS_PATH ]]; then
        echo "1";
    else
        set +e
        hadoop fs -test -e $ABS_PATH 1>/dev/null 2>&1
        if [[ $? -eq 0 ]]; then
            echo "1";
        else
            echo "0";
        fi
        set -e
    fi
}

function copy(){
    # from_file, to_file
    if [[ $DATA_PATH == hdfs://* ]]; then
         hadoop fs -put -f $1 $2
    else
        cp $1 $2
    fi
}


if [[ $(fileExists $TRANSCRIPT_PATH "train.en.txt") -eq 0 ]]; then
    echo "Missing file: $TRANSCRIPT_PATH/train.en.txt"
    exit 1
fi

if [[ $(fileExists $TRANSCRIPT_PATH "train.${TRG_LANGUAGE}.txt") -eq 0 ]]; then
    echo "Missing file: $TRANSCRIPT_PATH/train.${TRG_LANGUAGE}.txt"
    exit 1
fi


ASRST_OUTPUT_PATH=$DATA_PATH/asr_st/${TRG_LANGUAGE}
MT_OUTPUT_PATH=$DATA_PATH/mt/${TRG_LANGUAGE}
makeDirs $ASRST_OUTPUT_PATH/train
makeDirs $MT_OUTPUT_PATH/train

if [[ $DATA_PATH == hdfs://* ]]; then
    TRANSCRIPT_PATH=$THIS_DIR/transcripts
    rm -rf $TRANSCRIPT_PATH/
    makeDirs $TRANSCRIPT_PATH
    hadoop fs -get $DATA_PATH/transcripts/$TRG_LANGUAGE/train.en.txt $TRANSCRIPT_PATH/
    hadoop fs -get $DATA_PATH/transcripts/$TRG_LANGUAGE/train.$TRG_LANGUAGE.txt $TRANSCRIPT_PATH/
fi

echo "Remove punctuations and lowercase for en side"
python3 -c """
import tensorflow as tf
from neurst.utils.misc import PseudoPool
from neurst.data.data_pipelines.data_pipeline import lowercase_and_remove_punctuations


def apply_fn(sent_list):
    return [lowercase_and_remove_punctuations(
        '$TRG_LANGUAGE', line, lowercase=True, remove_punctuation=True) for line in sent_list]


threads = 10

with PseudoPool(threads) as process_pool:
    with tf.io.gfile.GFile('$TRANSCRIPT_PATH/train.en.txt') as fp:
        sentences = [line.strip() for line in fp]
    sents_per_thread = len(sentences) // threads
    sentences_list = []
    for idx in range(threads):
        sentences_list.append((
            sentences[idx * sents_per_thread:] if idx == threads - 1 else
            sentences[idx * sents_per_thread: (idx + 1) * sents_per_thread]))
    processed_list = process_pool.map(apply_fn, sentences_list)
with tf.io.gfile.GFile('$TRANSCRIPT_PATH/train.en.clean.txt', 'w') as fw:
    for line in sum(processed_list, []):
        fw.write(line.strip() + '\n')
"""

echo "tokenize data..."
perl $MOSES_TOKENIZER -l en -a -no-escape -threads 10 \
    < $TRANSCRIPT_PATH/train.en.clean.txt > $TRANSCRIPT_PATH/train.en.clean.tok.txt &
perl $MOSES_TOKENIZER -l $TRG_LANGUAGE -a -no-escape -threads 10 \
    < $TRANSCRIPT_PATH/train.$TRG_LANGUAGE.txt > $TRANSCRIPT_PATH/train.$TRG_LANGUAGE.tok.txt &
wait

echo "learn 8k BPE for ASR/MT/ST..."
subword-nmt learn-joint-bpe-and-vocab \
    --input $TRANSCRIPT_PATH/train.en.clean.tok.txt \
    --symbols 8000 \
    --output $TRANSCRIPT_PATH/codes.bpe.en \
    --write-vocabulary $TRANSCRIPT_PATH/vocab.en &

subword-nmt learn-joint-bpe-and-vocab \
    --input $TRANSCRIPT_PATH/train.$TRG_LANGUAGE.tok.txt \
    --symbols 8000 \
    --output $TRANSCRIPT_PATH/codes.bpe.$TRG_LANGUAGE \
    --write-vocabulary $TRANSCRIPT_PATH/vocab.$TRG_LANGUAGE &
wait


echo "Preprocess for MT..."
subword-nmt apply-bpe --codes $TRANSCRIPT_PATH/codes.bpe.en \
    --vocabulary $TRANSCRIPT_PATH/vocab.en < $TRANSCRIPT_PATH/train.en.clean.tok.txt \
        > $TRANSCRIPT_PATH/train.en.clean.tok.bpe.txt &

subword-nmt apply-bpe --codes $TRANSCRIPT_PATH/codes.bpe.$TRG_LANGUAGE \
    --vocabulary $TRANSCRIPT_PATH/vocab.$TRG_LANGUAGE < $TRANSCRIPT_PATH/train.$TRG_LANGUAGE.tok.txt \
        > $TRANSCRIPT_PATH/train.$TRG_LANGUAGE.tok.bpe.txt &
wait

get_seeded_random()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}

random_source=`date +%N`
shuf --random-source=<(get_seeded_random $random_source) \
    $TRANSCRIPT_PATH/train.en.clean.tok.bpe.txt > $TRANSCRIPT_PATH/train.en.clean.tok.bpe.txt.shuf
shuf --random-source=<(get_seeded_random $random_source) \
    $TRANSCRIPT_PATH/train.$TRG_LANGUAGE.tok.bpe.txt > $TRANSCRIPT_PATH/train.$TRG_LANGUAGE.tok.bpe.txt.shuf


echo "Preprocess for ASR/ST..."
rm -f FAILED

PROCESSORS_IN_PARALLEL=4
NUM_PROCESSORS=16
TOTAL_SHARDS=128
SHARD_PER_PROCESS=$((TOTAL_SHARDS / NUM_PROCESSORS))
LOOP=$((NUM_PROCESSORS / PROCESSORS_IN_PARALLEL))

for loopid in $(seq 1 ${LOOP}); do
    start=$(($((loopid - 1)) * ${PROCESSORS_IN_PARALLEL}))
    end=$(($start + PROCESSORS_IN_PARALLEL - 1))
    echo $start, $end
    for procid in $(seq $start $end); do
        set -x
        nice -n 10 python3 -m neurst.cli.create_tfrecords \
            --processor_id $procid --num_processors $NUM_PROCESSORS \
            --num_output_shards $TOTAL_SHARDS \
            --output_range_begin "$((SHARD_PER_PROCESS * procid))" \
            --output_range_end "$((SHARD_PER_PROCESS * procid + SHARD_PER_PROCESS))" \
        --dataset AudioTripleTFRecordDataset --feature_key "audio" \
        --transcript_key "transcript" --translation_key "translation" \
        --data_path $DATA_PATH/train/${TRG_LANGUAGE} \
        --output_template $ASRST_OUTPUT_PATH/train/train.tfrecords-%5.5d-of-%5.5d \
        --task MultiTaskSpeechTranslation \
        --task.params "
            transcript_data_pipeline.class: TranscriptDataPipeline
            transcript_data_pipeline.params:
                remove_punctuation: True
                lowercase: True
                language: en
                tokenizer: moses
                subtokenizer: bpe
                subtokenizer_codes: $TRANSCRIPT_PATH/codes.bpe
                vocab_path: $TRANSCRIPT_PATH/vocab.en
            translation_data_pipeline.class: TranscriptDataPipeline
            translation_data_pipeline.params:
                remove_punctuation: False
                lowercase: False
                language: $TRG_LANGUAGE
                tokenizer: moses
                subtokenizer: bpe
                subtokenizer_codes: $TRANSCRIPT_PATH/codes.bpe
                vocab_path: $TRANSCRIPT_PATH/vocab.$TRG_LANGUAGE" || touch FAILED &
        set +x
    done
    wait
    ! [[ -f FAILED ]]
done

copy $TRANSCRIPT_PATH/codes.bpe.en $MT_OUTPUT_PATH/
copy $TRANSCRIPT_PATH/codes.bpe.$TRG_LANGUAGE $MT_OUTPUT_PATH/
copy $TRANSCRIPT_PATH/vocab.en $MT_OUTPUT_PATH/
copy $TRANSCRIPT_PATH/vocab.$TRG_LANGUAGE $MT_OUTPUT_PATH/
copy $TRANSCRIPT_PATH/codes.bpe.en $ASRST_OUTPUT_PATH/
copy $TRANSCRIPT_PATH/vocab.en $ASRST_OUTPUT_PATH/
copy $TRANSCRIPT_PATH/vocab.$TRG_LANGUAGE $ASRST_OUTPUT_PATH/
copy $TRANSCRIPT_PATH/codes.bpe.$TRG_LANGUAGE $ASRST_OUTPUT_PATH/
copy $TRANSCRIPT_PATH/train.en.clean.tok.bpe.txt.shuf $MT_OUTPUT_PATH/train/train.en.clean.tok.bpe.txt
copy $TRANSCRIPT_PATH/train.$TRG_LANGUAGE.tok.bpe.txt.shuf $MT_OUTPUT_PATH/train/train.$TRG_LANGUAGE.tok.bpe.txt

if [[ $DATA_PATH == hdfs://* ]]; then
    rm -r $TRANSCRIPT_PATH
else
    rm $TRANSCRIPT_PATH/codes.* $TRANSCRIPT_PATH/vocab* $TRANSCRIPT_PATH/train.en.clean* $TRANSCRIPT_PATH/train.$TRG_LANGUAGE.tok*
fi

sed "s#DATA_PATH#${DATA_PATH}#" ${THIS_DIR}/asr_training_args.yml > _tmp_training
sed -i "s#TRG_LANG#${TRG_LANGUAGE}#" _tmp_training
sed -i "s#codes.bpe#codes.bpe.en#" _tmp_training
sed "s#DATA_PATH#${DATA_PATH}#" ${THIS_DIR}/asr_validation_args.yml > _tmp_validation
sed -i "s#TRG_LANG#${TRG_LANGUAGE}#" _tmp_validation
sed "s#DATA_PATH#${DATA_PATH}#" ${THIS_DIR}/asr_prediction_args.yml > _tmp_prediction
sed -i "s#TRG_LANG#${TRG_LANGUAGE}#" _tmp_prediction
copy _tmp_training $ASRST_OUTPUT_PATH/asr_training_args.yml
copy _tmp_validation $ASRST_OUTPUT_PATH/asr_validation_args.yml
copy _tmp_prediction $ASRST_OUTPUT_PATH/asr_prediction_args.yml

sed "s#DATA_PATH#${DATA_PATH}#" ${THIS_DIR}/mt_training_args.yml > _tmp_training
sed -i "s#TRG_LANG#${TRG_LANGUAGE}#g" _tmp_training
sed -i -e "0,/codes.bpe/{s/codes.bpe/codes.bpe.en/}" _tmp_training
sed -i "s#codes.bpe#codes.bpe.${TRG_LANGUAGE}#g" _tmp_training
sed -i "s#codes.bpe.${TRG_LANGUAGE}.en#codes.bpe.en#" _tmp_training
sed "s#DATA_PATH#${DATA_PATH}#" ${THIS_DIR}/mt_validation_args.yml > _tmp_validation
sed -i "s#TRG_LANG#${TRG_LANGUAGE}#g" _tmp_validation
sed "s#DATA_PATH#${DATA_PATH}#" ${THIS_DIR}/mt_prediction_args.yml > _tmp_prediction
sed -i "s#TRG_LANG#${TRG_LANGUAGE}#g" _tmp_prediction
copy _tmp_training $MT_OUTPUT_PATH/mt_training_args.yml
copy _tmp_validation $MT_OUTPUT_PATH/mt_validation_args.yml
copy _tmp_prediction $MT_OUTPUT_PATH/mt_prediction_args.yml

sed "s#DATA_PATH#${DATA_PATH}#" ${THIS_DIR}/st_training_args.yml > _tmp_training
sed -i "s#TRG_LANG#${TRG_LANGUAGE}#g" _tmp_training
sed -i "s#codes.bpe#codes.bpe.${TRG_LANGUAGE}#g" _tmp_training
sed "s#DATA_PATH#${DATA_PATH}#" ${THIS_DIR}/st_validation_args.yml > _tmp_validation
sed -i "s#TRG_LANG#${TRG_LANGUAGE}#g" _tmp_validation
sed "s#DATA_PATH#${DATA_PATH}#" ${THIS_DIR}/st_prediction_args.yml > _tmp_prediction
sed -i "s#TRG_LANG#${TRG_LANGUAGE}#g" _tmp_prediction
copy _tmp_training $ASRST_OUTPUT_PATH/st_training_args.yml
copy _tmp_validation $ASRST_OUTPUT_PATH/st_validation_args.yml
copy _tmp_prediction $ASRST_OUTPUT_PATH/st_prediction_args.yml

rm _tmp_training _tmp_validation _tmp_prediction

