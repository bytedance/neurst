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
    echo "Usage: ./03-preprocess.sh MOSES_DIR DATA_PATH"
    exit 1
fi

if [[ ! -n "$2" ]] ;then
    echo "Usage: ./03-preprocess_mt_st.sh MOSES_DIR DATA_PATH"
    exit 1
else
    DATA_PATH="$2"
fi

TRANSCRIPT_PATH=$DATA_PATH/transcripts

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

if [[ $(fileExists $TRANSCRIPT_PATH "train.fr.txt") -eq 0 ]]; then
    echo "Missing file: $TRANSCRIPT_PATH/train.fr.txt"
    exit 1
fi

ASRST_OUTPUT_PATH=$DATA_PATH/asr_st
MT_OUTPUT_PATH=$DATA_PATH/mt
makeDirs $ASRST_OUTPUT_PATH/train
makeDirs $MT_OUTPUT_PATH/train

if [[ $DATA_PATH == hdfs://* ]]; then
    TRANSCRIPT_PATH=$THIS_DIR/transcripts
    rm -rf $TRANSCRIPT_PATH/
    makeDirs $TRANSCRIPT_PATH
    hadoop fs -get $DATA_PATH/transcripts/train.en.txt $TRANSCRIPT_PATH/
    hadoop fs -get $DATA_PATH/transcripts/train.fr.txt $TRANSCRIPT_PATH/
fi

echo "tokenize the target side and learn joint 8k BPE..."
perl $MOSES_TOKENIZER -l fr -a -no-escape -threads 10 < $TRANSCRIPT_PATH/train.fr.txt > $TRANSCRIPT_PATH/train.fr.tok.txt

subword-nmt learn-joint-bpe-and-vocab \
    --input $TRANSCRIPT_PATH/train.en.txt $TRANSCRIPT_PATH/train.fr.tok.txt \
    --symbols 8000 \
    --output $TRANSCRIPT_PATH/codes.bpe \
    --write-vocabulary $TRANSCRIPT_PATH/vocab.en $TRANSCRIPT_PATH/vocab.fr

subword-nmt apply-bpe --codes $TRANSCRIPT_PATH/codes.bpe \
    --vocabulary $TRANSCRIPT_PATH/vocab.en < $TRANSCRIPT_PATH/train.en.txt \
        > $TRANSCRIPT_PATH/train.en.bpe.txt &

subword-nmt apply-bpe --codes $TRANSCRIPT_PATH/codes.bpe \
    --vocabulary $TRANSCRIPT_PATH/vocab.fr  < $TRANSCRIPT_PATH/train.fr.tok.txt \
        > $TRANSCRIPT_PATH/train.fr.tok.bpe.txt &
wait

get_seeded_random()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}

random_source=`date +%N`
shuf --random-source=<(get_seeded_random $random_source) \
    $TRANSCRIPT_PATH/train.en.bpe.txt > $TRANSCRIPT_PATH/train.en.bpe.txt.shuf
shuf --random-source=<(get_seeded_random $random_source) \
    $TRANSCRIPT_PATH/train.fr.tok.bpe.txt > $TRANSCRIPT_PATH/train.fr.tok.bpe.txt.shuf

echo "=== Generating TF Records projected transcripts ==="
rm -f FAILED
SERIES=4
SHARDS=8
SERIES_PER_SHARD=$((${SHARDS} / ${SERIES}))

for series in $(seq 1 ${SERIES}); do
    start=$(($(($series - 1)) * ${SERIES_PER_SHARD}))
    end=$(($start + $SERIES_PER_SHARD - 1))
    for subshard in $(seq $start $end); do
        set -x
        nice -n 10 python3 -m neurst.cli.create_tfrecords \
            --processor_id ${subshard} --num_processors 8 \
            --num_output_shards 64 \
            --output_range_begin "$((8 * subshard))" \
            --output_range_end "$((8 * subshard + 8))" \
            --dataset AudioTripleTFRecordDataset --feature_key "audio" \
            --transcript_key "transcript" --translation_key "translation" \
            --data_path $DATA_PATH/train \
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
                    language: fr
                    tokenizer: moses
                    subtokenizer: bpe
                    subtokenizer_codes: $TRANSCRIPT_PATH/codes.bpe
                    vocab_path: $TRANSCRIPT_PATH/vocab.fr" || touch FAILED &
        set +x
    done
    wait
    ! [[ -f FAILED ]]
done

copy $TRANSCRIPT_PATH/codes.bpe $MT_OUTPUT_PATH/
copy $TRANSCRIPT_PATH/codes.bpe $ASRST_OUTPUT_PATH/
copy $TRANSCRIPT_PATH/vocab.fr $MT_OUTPUT_PATH/
copy $TRANSCRIPT_PATH/vocab.en $MT_OUTPUT_PATH/
copy $TRANSCRIPT_PATH/vocab.fr $ASRST_OUTPUT_PATH/
copy $TRANSCRIPT_PATH/vocab.en $ASRST_OUTPUT_PATH/
copy $TRANSCRIPT_PATH/train.en.bpe.txt.shuf $MT_OUTPUT_PATH/train/train.en.bpe.txt
copy $TRANSCRIPT_PATH/train.fr.tok.bpe.txt.shuf $MT_OUTPUT_PATH/train/train.fr.tok.bpe.txt

sed "s#DATA_PATH#${DATA_PATH}#" ${THIS_DIR}/mt_training_args.yml > _training
sed "s#DATA_PATH#${DATA_PATH}#" ${THIS_DIR}/mt_validation_args.yml > _validation
sed "s#DATA_PATH#${DATA_PATH}#" ${THIS_DIR}/mt_prediction_args.yml > _prediction
copy _training $MT_OUTPUT_PATH/mt_training_args.yml
copy _validation $MT_OUTPUT_PATH/mt_validation_args.yml
copy _prediction $MT_OUTPUT_PATH/mt_prediction_args.yml

sed "s#DATA_PATH#${DATA_PATH}#" ${THIS_DIR}/asr_training_args.yml > _training
sed "s#DATA_PATH#${DATA_PATH}#" ${THIS_DIR}/asr_validation_args.yml > _validation
sed "s#DATA_PATH#${DATA_PATH}#" ${THIS_DIR}/asr_prediction_args.yml > _prediction
copy _training $ASRST_OUTPUT_PATH/asr_training_args.yml
copy _validation $ASRST_OUTPUT_PATH/asr_validation_args.yml
copy _prediction $ASRST_OUTPUT_PATH/asr_prediction_args.yml

sed "s#DATA_PATH#${DATA_PATH}#" ${THIS_DIR}/st_training_args.yml > _training
sed "s#DATA_PATH#${DATA_PATH}#" ${THIS_DIR}/st_validation_args.yml > _validation
sed "s#DATA_PATH#${DATA_PATH}#" ${THIS_DIR}/st_prediction_args.yml > _prediction
copy _training $ASRST_OUTPUT_PATH/st_training_args.yml
copy _validation $ASRST_OUTPUT_PATH/st_validation_args.yml
copy _prediction $ASRST_OUTPUT_PATH/st_prediction_args.yml

rm _training _validation _prediction

if [[ $DATA_PATH == hdfs://* ]]; then
    rm -r $TRANSCRIPT_PATH
else
    rm $TRANSCRIPT_PATH/codes.*
    rm $TRANSCRIPT_PATH/vocab.*
    rm $TRANSCRIPT_PATH/train.en.bpe*
    rm $TRANSCRIPT_PATH/train.fr.tok*
fi
