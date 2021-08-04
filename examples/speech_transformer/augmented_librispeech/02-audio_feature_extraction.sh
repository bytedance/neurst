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

if [[ ! -n "$1" ]] ;then
    echo "Usage: ./02-audio_feature_extraction.sh ROOT_DATA_PATH"
    exit 1;
else
    DATA_PATH="$1"
fi

RAW_DATA_PATH=$DATA_PATH/raw
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

makeDirs $TRANSCRIPT_PATH

if [[ $(fileExists $RAW_DATA_PATH "train_100h.zip") -eq 0 ]]; then
    echo "File not exists: $RAW_DATA_PATH/train_100h.zip"
    echo "Please download and save it to $RAW_DATA_PATH in advance"
    exit 1
fi


if [[ $(fileExists $RAW_DATA_PATH "dev.zip") -eq 0 ]]; then
    echo "File not exists: $RAW_DATA_PATH/dev.zip"
    echo "Please download and save it to $RAW_DATA_PATH in advance"
    exit 1
fi

if [[ $(fileExists $RAW_DATA_PATH "test.zip") -eq 0 ]]; then
    echo "File not exists: $RAW_DATA_PATH/test.zip"
    echo "Please download and save it to $RAW_DATA_PATH in advance"
    exit 1
fi

echo "=== First pass, collecting transcripts ==="

set -x
python3 -m neurst.cli.extract_audio_transcripts \
    --dataset AugmentedLibriSpeech \
    --input_tarball $RAW_DATA_PATH/train_100h.zip \
    --output_transcript_file $TRANSCRIPT_PATH/train.en.txt \
    --output_translation_file  $TRANSCRIPT_PATH/train.fr.txt

python3 -m neurst.cli.extract_audio_transcripts \
    --dataset AugmentedLibriSpeech \
    --input_tarball $RAW_DATA_PATH/dev.zip \
    --output_transcript_file $TRANSCRIPT_PATH/dev.en.txt \
    --output_translation_file  $TRANSCRIPT_PATH/dev.fr.txt

python3 -m neurst.cli.extract_audio_transcripts \
    --dataset AugmentedLibriSpeech \
    --input_tarball $RAW_DATA_PATH/test.zip \
    --output_transcript_file $TRANSCRIPT_PATH/test.en.txt \
    --output_translation_file  $TRANSCRIPT_PATH/test.fr.txt
set +x

echo "=== Second pass, generating TF Records with audio features and raw transcripts ==="
makeDirs $DATA_PATH/train
rm -f FAILED

PROCESSORS_IN_PARALLEL=4
NUM_PROCESSORS=8
TOTAL_SHARDS=64
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
        --dataset AugmentedLibriSpeech \
        --feature_extractor.class fbank \
        --feature_extractor.params '{"nfilt": 80}' \
        --input_tarball $RAW_DATA_PATH/train_100h.zip \
        --output_template $DATA_PATH/train/train.tfrecords-%5.5d-of-%5.5d || touch FAILED &
        set +x
    done
    wait
    ! [[ -f FAILED ]]
done

makeDirs $DATA_PATH/devtest
for subset in dev test; do
    set -x
    nice -n 10 python3 -m neurst.cli.create_tfrecords \
        --processor_id 0 --num_processors 1 \
        --num_output_shards 1 \
        --output_range_begin 0 \
        --output_range_end 1 \
    --dataset AugmentedLibriSpeech \
    --feature_extractor.class fbank \
    --feature_extractor.params '{"nfilt": 80}' \
    --input_tarball $RAW_DATA_PATH/${subset}.zip \
    --output_template $DATA_PATH/devtest/${subset}.tfrecords-%5.5d-of-%5.5d || touch FAILED &
    set +x
done
wait
! [[ -f FAILED ]]



