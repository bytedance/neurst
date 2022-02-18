#!/usr/bin/env bash

set -e

DATA_PATH=$1

if [[ -z $DATA_PATH ]]; then
    echo "Usage: ./03-create_devtest_set.sh DATA_PATH"
    exit 1;
fi

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

function makeDirs(){
    if [[ $1 == hdfs://* ]]; then
        hadoop fs -mkdir -p $1
    else
        mkdir -p $1
    fi
}

ASR_OUTPUT_PATH=$DATA_PATH/asr
makeDirs $ASR_OUTPUT_PATH/devtest

echo "Creating TFRecord for dev test set..."
rm -f FAILED
for subset in DEV TEST; do
    set -x
    nice -n 10 python3 -m neurst.cli.create_tfrecords \
        --processor_id 0 --num_processors 1 \
        --num_output_shards 1 --output_range_begin 0 --output_range_end 1 \
        --output_template $ASR_OUTPUT_PATH/devtest/$subset.tfrecords-%5.5d-of-%5.5d \
        --dataset.class GigaSpeech \
        --input_tarball $DATA_PATH \
        --subset $subset \
        --feature_extractor.class fbank \
        --feature_extractor.params '{"nfilt":80}' || touch FAILED &
    set +x
done
wait
! [[ -f FAILED ]]

function copy(){
    # from_file, to_file
    if [[ $DATA_PATH == hdfs://* ]]; then
         hadoop fs -put -f $1 $2
    else
        cp $1 $2
    fi
}

THIS_DIR="$( cd "$( dirname "$0" )" && pwd )"
sed "s#DATA_PATH#${DATA_PATH}#" $THIS_DIR/asr_validation_args.yml > $THIS_DIR/_tmp_valid
copy $THIS_DIR/_tmp_valid $ASR_OUTPUT_PATH/asr_validation_args.yml

sed "s#DATA_PATH#${DATA_PATH}#" $THIS_DIR/asr_prediction_args.yml > $THIS_DIR/_tmp_predict
copy $THIS_DIR/_tmp_predict $ASR_OUTPUT_PATH/asr_prediction_args.yml

rm $THIS_DIR/_tmp_valid
rm $THIS_DIR/_tmp_predict
