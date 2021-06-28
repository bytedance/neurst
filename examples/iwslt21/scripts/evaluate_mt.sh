#!/usr/bin/env bash

set -e

if [[ ! -n "$3" ]] ;then
    echo "Usage: ./evaluate_mt.sh TEST_SET MODEL_DIR OUTPUT_PATH"
    exit 1;
fi

TEST_SET=$1
MODEL_DIR=$2
OUTPUT_PATH=$3

URL_PREFIX="http://sf3-ttcdn-tos.pstatp.com/obj/nlp-opensource/neurst/iwslt21/offline"
CFG_URL="${URL_PREFIX}/cfgs/mt_prediction_args.yml"
DATA_URL_PREFIX="${URL_PREFIX}/devtests"

case $TEST_SET in
    "mustc-v1-dev")
    SRC_FILE="mustc_v1.0_en-de.dev.tagen.txt"
    TRG_FILE="mustc_v1.0_en-de.dev.de.txt"
    ;;

    "mustc-v1-tst")
    SRC_FILE="mustc_v1.0_en-de.tst-COMMON.tagen.txt"
    TRG_FILE="mustc_v1.0_en-de.tst-COMMON.de.txt"
    ;;

    "mustc-v2-dev")
    SRC_FILE="mustc_v2.0_en-de.dev.tagen.txt"
    TRG_FILE="mustc_v2.0_en-de.dev.de.txt"
    ;;

    "mustc-v2-tst")
    SRC_FILE="mustc_v2.0_en-de.tst-COMMON.tagen.txt"
    TRG_FILE="mustc_v2.0_en-de.tst-COMMON.de.txt"
    ;;

    "mustc-v1-dev-tc")
    SRC_FILE="mustc_v1.0_en-de.dev.en.txt"
    TRG_FILE="mustc_v1.0_en-de.dev.de.txt"
    ;;

    "mustc-v1-tst-tc")
    SRC_FILE="mustc_v1.0_en-de.tst-COMMON.en.txt"
    TRG_FILE="mustc_v1.0_en-de.tst-COMMON.de.txt"
    ;;

    "mustc-v2-dev-tc")
    SRC_FILE="mustc_v2.0_en-de.dev.en.txt"
    TRG_FILE="mustc_v2.0_en-de.dev.de.txt"
    ;;

    "mustc-v2-tst-tc")
    SRC_FILE="mustc_v2.0_en-de.tst-COMMON.en.txt"
    TRG_FILE="mustc_v2.0_en-de.tst-COMMON.de.txt"
    ;;

    *)  echo "Unknown ${TEST_SET}"
    ;;
esac

TST_FILE_PREFIX=${SRC_FILE%.*}
TST_FILE_PREFIX=${TST_FILE_PREFIX%.*}
OUTPUT_MT_FILE=$OUTPUT_PATH/$TST_FILE_PREFIX.de.hypo.txt
OUTPUT_MT_NOTAG_FILE=$OUTPUT_PATH/$TST_FILE_PREFIX.de.hypo.notag.txt
LOCAL_REF_FILE=$OUTPUT_PATH/$TST_FILE_PREFIX.de.ref.txt
LOCAL_REF_NOTAG_FILE=$OUTPUT_PATH/$TST_FILE_PREFIX.de.ref.notag.txt

curl $DATA_URL_PREFIX/$TRG_FILE -o $LOCAL_REF_FILE

# inference
python3 -m neurst.cli.run_exp --config_paths $CFG_URL --src_file $DATA_URL_PREFIX/$SRC_FILE --output_file $OUTPUT_MT_FILE --model_dir $MODEL_DIR 1>/dev/null 2>&1

# clean hypothesis
perl -pe 's/\([^\)]+\)//g;' $LOCAL_REF_FILE | tr -s " " > $LOCAL_REF_NOTAG_FILE
perl -pe 's/\([^\)]+\)//g;' $OUTPUT_MT_FILE | tr -s " " > $OUTPUT_MT_NOTAG_FILE


echo "============================= Evaluation (including tags) =============================" > $OUTPUT_PATH/$TST_FILE_PREFIX.bleu.txt
cat $OUTPUT_MT_FILE | sacrebleu -l en-de $LOCAL_REF_FILE >> $OUTPUT_PATH/$TST_FILE_PREFIX.bleu.txt
echo "" >> $OUTPUT_PATH/$TST_FILE_PREFIX.bleu.txt

echo "============================= Evaluation (no tags) =============================" >> $OUTPUT_PATH/$TST_FILE_PREFIX.bleu.txt
cat $OUTPUT_MT_NOTAG_FILE | sacrebleu -l en-de $LOCAL_REF_NOTAG_FILE >> $OUTPUT_PATH/$TST_FILE_PREFIX.bleu.txt
