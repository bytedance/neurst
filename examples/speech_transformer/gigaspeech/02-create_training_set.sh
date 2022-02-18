#!/usr/bin/env bash

set -e

REMOVE_PUNCTUATION=True

until [[ -z $1 ]]
do
    case $1 in
        --keep-punctuation)
            shift; REMOVE_PUNCTUATION=False;;
        *)
            if [[ -z $DATA_PATH ]]; then
                DATA_PATH=$1;
            elif [[ -z $SUBSET ]]; then
                SUBSET=$1;
            fi
            shift;;
    esac
done

if [[ -z $DATA_PATH ]] || [[ -z $SUBSET ]]; then
    echo "Usage: ./02-create_training_set.sh DATA_PATH SUBSET (--keep-punctuation)"
    exit 1;
fi

SUBSETS="XL L M S XS"
if [[ ! $SUBSETS =~ $SUBSET ]]; then
    echo "${SUBSET} not supported. Please provide a subset in ${SUBSETS}"
    exit 1
fi
echo "Extract subset ${SUBSET}."

THIS_DIR="$( cd "$( dirname "$0" )" && pwd )"

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

# check meta file
if [[ $(fileExists $DATA_PATH GigaSpeech.json) -eq 1 ]]; then
    META_FILE=$DATA_PATH/GigaSpeech.json
else
    echo "GigaSpeech.json does not exist at ${DATA_PATH}."
    exit 1
fi

echo "Collecting transcript from ${META_FILE}..."
LOCAL_TRANSCRIPT=$THIS_DIR/text_all

python3 -c """
import json
import os

import tensorflow as tf

repl_marks_w_punc = [['<QUESTIONMARK>', '?'], ['<EXCLAMATIONPOINT>', '!'],
                     ['<PERIOD>', '.'], ['<COMMA>', ','],
                     [' ?', '?'], [' !', '!'], [' .', '.'], [' ,', ',']]

repl_marks_wo_punc = [['<QUESTIONMARK>', ''], ['<EXCLAMATIONPOINT>', ''],
                      ['<PERIOD>', ''], ['<COMMA>', ''],
                      ['  ', ' ']]

with tf.io.gfile.GFile('${META_FILE}') as fp, \
    open('${LOCAL_TRANSCRIPT}',mode='w+') as transcript:
    meta = json.load(fp)
    audios = meta['audios']
    for audio in audios:
        segments = audio['segments']
        for segment in segments:
            if '{TEST}' in segment['subsets'] or '{DEV}' in segment['subsets']:
                continue
            text = segment['text_tn']
            if '<SIL>' in text or '<NOISE>' in text or '<MUSIC>' in text or '<OTHER>' in text:
                continue
            if ${REMOVE_PUNCTUATION}:
                for ori, rpl in repl_marks_wo_punc:
                    text = text.replace(ori, rpl)
            else:
                for ori, rpl in repl_marks_w_punc:
                    text = text.replace(ori, rpl)
            text = text.lower()
            transcript.write(f'{text}\n')
"""

echo "Learning spm vocabulary and model..."
python3 -c """
import sentencepiece as spm
spm.SentencePieceTrainer.train(input='${LOCAL_TRANSCRIPT}', model_prefix='${THIS_DIR}/spm', 
    vocab_size=10000,character_coverage=1.0, model_type='unigram')
"""

rm $LOCAL_TRANSCRIPT

function copy(){
    # from_file, to_file
    if [[ $DATA_PATH == hdfs://* ]]; then
         hadoop fs -put -f $1 $2
    else
        cp $1 $2
    fi
}

function makeDirs(){
    if [[ $1 == hdfs://* ]]; then
        hadoop fs -mkdir -p $1
    else
        mkdir -p $1
    fi
}

ASR_OUTPUT_PATH=$DATA_PATH/asr
makeDirs $ASR_OUTPUT_PATH/train

sed "s#DATA_PATH#${DATA_PATH}#" $THIS_DIR/asr_data_prep.yml > $THIS_DIR/_tmp_prep
sed -i "s#SUBSET#${SUBSET}#g" $THIS_DIR/_tmp_prep
sed -i "s#REMOVE_PUNCTUATION#${REMOVE_PUNCTUATION}#g" $THIS_DIR/_tmp_prep
copy $THIS_DIR/_tmp_prep $ASR_OUTPUT_PATH/asr_data_prep.yml
rm $THIS_DIR/_tmp_prep

copy $THIS_DIR/spm.model $DATA_PATH/spm.model
copy $THIS_DIR/spm.vocab $DATA_PATH/spm.vocab
rm $THIS_DIR/spm.model
rm $THIS_DIR/spm.vocab

rm -f FAILED
# XS only has 1 shard
if [[ $SUBSET == "XS" ]]; then
    set +x
    nice -n 10 python3 -m neurst.cli.create_tfrecords \
        --processor_id 0 --num_processors 1 \
        --num_output_shards 1 \
        --output_range_begin 0 \
        --output_range_end 1 \
        --output_template $ASR_OUTPUT_PATH/train/train.tfrecords-%5.5d-of-%5.5d \
        --config_paths $ASR_OUTPUT_PATH/asr_data_prep.yml \
        --progressbar
    set -x
# large subsets have several shards
else
    PROCESSORS_IN_PARALLEL=8
    NUM_PROCESSORS=32
    if [[ $SUBSET == "XL" ]]; then
        TOTAL_SHARDS=1664
    elif [[ $SUBSET == "L" ]]; then
        TOTAL_SHARDS=416
    elif [[ $SUBSET == "M" ]]; then
        TOTAL_SHARDS=160
    else
        # at least the same as NUM_PROCESSORS
        TOTAL_SHARDS=64
    fi
    SHARD_PER_PROCESS=$((TOTAL_SHARDS / NUM_PROCESSORS))
    LOOP=$((NUM_PROCESSORS / PROCESSORS_IN_PARALLEL))

    echo "Creating TFRecord for training set..."
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
                --output_template $ASR_OUTPUT_PATH/train/train.tfrecords-%5.5d-of-%5.5d \
                --config_paths $ASR_OUTPUT_PATH/asr_data_prep.yml \
                --progressbar || touch FAILED &
            set +x
        done
        wait
        ! [[ -f FAILED ]]
    done
fi

sed "s#DATA_PATH#${DATA_PATH}#" $THIS_DIR/asr_training_args.yml > $THIS_DIR/_tmp_train
sed -i "s#REMOVE_PUNCTUATION#${REMOVE_PUNCTUATION}#g" $THIS_DIR/_tmp_train
copy $THIS_DIR/_tmp_train $ASR_OUTPUT_PATH/asr_training_args.yml

rm $THIS_DIR/_tmp_train
