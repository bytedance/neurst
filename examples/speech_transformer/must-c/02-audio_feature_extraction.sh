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

until [[ -z "$1" ]]
do
    case $1 in
        --untar)
            shift; UNTAR=1;;
        *)
            if [[ -z $DATA_PATH ]]; then
                DATA_PATH=$1;
            elif [[ -z $TRG_LANGUAGE ]]; then
                TRG_LANGUAGE=$1;
            fi
            shift;;
    esac
done

if [[ -z $DATA_PATH ]] || [[ -z $TRG_LANGUAGE ]];then
    echo "Usage: ./02-audio_feature_extraction.sh ROOT_DATA_PATH TRG_LANG (--untar)"
    exit 1;
fi

RAW_DATA_PATH=$DATA_PATH/raw/
TRANSCRIPT_PATH=$DATA_PATH/transcripts/${TRG_LANGUAGE}

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

if [[ $(fileExists $RAW_DATA_PATH "MUSTC_v1.0_en-${TRG_LANGUAGE}.tar.gz") -eq 0 ]]; then
    echo "File not exists: $RAW_DATA_PATH/MUSTC_v1.0_en-${TRG_LANGUAGE}.tar.gz"
    echo "Please download and save it to $RAW_DATA_PATH in advance"
    exit 1
fi

INPUT_TARBALL=$RAW_DATA_PATH/"MUSTC_v1.0_en-${TRG_LANGUAGE}.tar.gz"
echo "Extract from $INPUT_TARBALL..."

INPUT_FILE=$INPUT_TARBALL

if [[ ! -z $UNTAR ]]; then
    echo "Untar..."
    python3 -c """
import tensorflow as tf
import tarfile

def non_h5(members):
    for tarinfo in members:
        if not tarinfo.name.endswith('.h5'):
            yield tarinfo

with tf.io.gfile.GFile('$INPUT_TARBALL', 'rb') as fp:
    with tarfile.open(fileobj=fp, mode='r:*') as tar:
        tar.extractall(members=non_h5(tar))
    """
    INPUT_FILE="en-${TRG_LANGUAGE}"
fi

echo "=== First pass, collecting transcripts ==="

set -x
python3 -m neurst.cli.extract_audio_transcripts \
    --dataset MuSTC --extraction train \
    --input_tarball $INPUT_FILE \
    --output_transcript_file $TRANSCRIPT_PATH/train.en.txt \
    --output_translation_file  $TRANSCRIPT_PATH/train.${TRG_LANGUAGE}.txt &

python3 -m neurst.cli.extract_audio_transcripts \
    --dataset MuSTC --extraction dev \
    --input_tarball $INPUT_FILE \
    --output_transcript_file $TRANSCRIPT_PATH/dev.en.txt \
    --output_translation_file  $TRANSCRIPT_PATH/dev.${TRG_LANGUAGE}.txt &

python3 -m neurst.cli.extract_audio_transcripts \
    --dataset MuSTC --extraction tst-COMMON \
    --input_tarball $INPUT_FILE \
    --output_transcript_file $TRANSCRIPT_PATH/tst-COMMON.en.txt \
    --output_translation_file  $TRANSCRIPT_PATH/tst-COMMON.${TRG_LANGUAGE}.txt &

wait
set +x

echo "=== Second pass, generating TF Records with audio features and raw transcripts ==="
makeDirs $DATA_PATH/train
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
        --dataset MuSTC --extraction "train" \
        --feature_extractor.class fbank \
        --feature_extractor.params '{"nfilt": 80}' \
        --input_tarball $INPUT_FILE \
        --output_template $DATA_PATH/train/${TRG_LANGUAGE}/train.tfrecords-%5.5d-of-%5.5d || touch FAILED &
        set +x
    done
    wait
    ! [[ -f FAILED ]]
done


makeDirs $DATA_PATH/devtest
for subset in dev tst-COMMON; do
    set -x
    nice -n 10 python3 -m neurst.cli.create_tfrecords \
        --processor_id 0 --num_processors 1 \
        --num_output_shards 1 \
        --output_range_begin 0 \
        --output_range_end 1 \
    --dataset MuSTC --extraction $subset \
    --feature_extractor.class fbank \
    --feature_extractor.params '{"nfilt": 80}' \
    --input_tarball $INPUT_FILE \
    --output_template $DATA_PATH/devtest/${subset}.en-${TRG_LANGUAGE}.tfrecords-%5.5d-of-%5.5d || touch FAILED &
    set +x
done
wait
! [[ -f FAILED ]]

if [[ ! -z $UNTAR ]]; then
    rm -r "en-${TRG_LANGUAGE}"
fi

