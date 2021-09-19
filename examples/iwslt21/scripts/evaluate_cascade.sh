#!/usr/bin/env bash

set -e

if [[ ! -n "$3" ]] ;then
    echo "Usage: ./evaluate_cascade.sh TEST_SET ASR_MODEL_DIR MT_MODEL_DIR OUTPUT_PATH"
    exit 1;
fi

TEST_SET=$1
ASR_MODEL_DIR=$2
MT_MODEL_DIR=$3
OUTPUT_PATH=$4

URL_PREFIX="http://sf3-ttcdn-tos.pstatp.com/obj/nlp-opensource/neurst/iwslt21/offline"
ASR_CFG_URL="${URL_PREFIX}/cfgs/asr_prediction_args.yml"
MT_CFG_URL="${URL_PREFIX}/cfgs/mt_prediction_args.yml"
DATA_URL_PREFIX="${URL_PREFIX}/devtests"

case $TEST_SET in
    "mustc-v1-dev")
    DATA_FILE="mustc_v1.0_en-de.dev.tfrecords-00000-of-00001"
    ;;

    "mustc-v1-tst")
    DATA_FILE="mustc_v1.0_en-de.tst-COMMON.tfrecords-00000-of-00001"
    ;;

    "mustc-v2-dev")
    DATA_FILE="mustc_v2.0_en-de.dev.tfrecords-00000-of-00001"
    ;;

    "mustc-v2-tst")
    DATA_FILE="mustc_v2.0_en-de.tst-COMMON.tfrecords-00000-of-00001"
    ;;

    "tst2020")
    DATA_FILE="iwslt-slt.tst2020.tfrecords-00000-of-00001"
    ;;

    "tst2021")
    DATA_FILE="iwslt-slt.tst2021.tfrecords-00000-of-00001"
    ;;

    *)  echo "Unknown ${TEST_SET}"
    ;;
esac

TST_FILE_PREFIX=${DATA_FILE%.*}
OUTPUT_ASR_FILE=$OUTPUT_PATH/$TST_FILE_PREFIX.en.hypo.txt
INPUT_ASR_FILE=$OUTPUT_PATH/$TST_FILE_PREFIX.en.tagasr.txt
OUTPUT_MT_FILE=$OUTPUT_PATH/$TST_FILE_PREFIX.de.hypo.txt
OUTPUT_MT_NOTAG_FILE=$OUTPUT_PATH/$TST_FILE_PREFIX.de.hypo.notag.txt
LOCAL_ASR_FILE=$OUTPUT_PATH/$TST_FILE_PREFIX.en.ref.txt
LOCAL_REF_FILE=$OUTPUT_PATH/$TST_FILE_PREFIX.de.ref.txt
LOCAL_REF_NOTAG_FILE=$OUTPUT_PATH/$TST_FILE_PREFIX.de.ref.notag.txt

# run ASR
python3 -m neurst.cli.run_exp --config_paths $ASR_CFG_URL --data_path $DATA_URL_PREFIX/$DATA_FILE --output_file $OUTPUT_ASR_FILE --model_dir $ASR_MODEL_DIR 1>/dev/null 2>&1

# add tag
sed 's/^/<ASR> /' $OUTPUT_ASR_FILE > $INPUT_ASR_FILE

# run MT
python3 -m neurst.cli.run_exp --config_paths $MT_CFG_URL --src_file $INPUT_ASR_FILE --output_file $OUTPUT_MT_FILE --model_dir $MT_MODEL_DIR 1>/dev/null 2>&1

if [[ $TEST_SET == tst20* ]];
then
    # clean hypothesis
    perl -pe 's/\([^\)]+\)//g;' $OUTPUT_MT_FILE | tr -s " " > $OUTPUT_MT_NOTAG_FILE
else
    python3 -c """
from neurst.data.datasets import build_dataset

ds = build_dataset({
    'class': 'AudioTripleTFRecordDataset',
    'params': {
        'data_path': '$DATA_URL_PREFIX/$DATA_FILE'
    }
})

with open('$LOCAL_ASR_FILE', 'w') as fw_asr, open('$LOCAL_REF_FILE', 'w') as fw_mt:
    for x in ds.build_iterator()():
        fw_asr.write(x['transcript']+'\n')
        fw_mt.write(x['translation']+'\n')
    """  1>/dev/null 2>&1

    # clean hypothesis
    perl -pe 's/\([^\)]+\)//g;' $LOCAL_REF_FILE | tr -s " " > $LOCAL_REF_NOTAG_FILE
    perl -pe 's/\([^\)]+\)//g;' $OUTPUT_MT_FILE | tr -s " " > $OUTPUT_MT_NOTAG_FILE

    python3 -m neurst.cli.text_metric --metric wer --hypo_file $OUTPUT_ASR_FILE --ref_file $LOCAL_ASR_FILE > $OUTPUT_PATH/$TST_FILE_PREFIX.bleu.txt 2>&1


    echo "============================= Evaluation (including tags) =============================" >> $OUTPUT_PATH/$TST_FILE_PREFIX.bleu.txt
    cat $OUTPUT_MT_FILE | sacrebleu -l en-de $LOCAL_REF_FILE >> $OUTPUT_PATH/$TST_FILE_PREFIX.bleu.txt
    echo "" >> $OUTPUT_PATH/$TST_FILE_PREFIX.bleu.txt

    echo "============================= Evaluation (no tags) =============================" >> $OUTPUT_PATH/$TST_FILE_PREFIX.bleu.txt
    cat $OUTPUT_MT_NOTAG_FILE | sacrebleu -l en-de $LOCAL_REF_NOTAG_FILE >> $OUTPUT_PATH/$TST_FILE_PREFIX.bleu.txt

fi
