#!/usr/bin/env bash
#
# Adapted from https://github.com/RayeRen/multilingual-kd-pytorch/blob/master/data/iwslt/raw/prepare-iwslt14.sh

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git


SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=subword-nmt
BPE_TOKENS=30000
prep=iwslt14.tokenized
tmp=$prep/tmp
orig=orig
rm -r $orig
rm -r $tmp
rm -r $prep
mkdir -p $orig $tmp $prep

for src in ar de es fa he it nl pl; do
    tgt=en
    lang=$src-en

    echo "pre-processing train data..."
    for l in $src $tgt; do
        if [[ ! -f $src-en.tgz ]]; then
            wget https://wit3.fbk.eu/archive/2014-01//texts/$src/en/$src-en.tgz
        fi
        cd $orig
        tar zxvf ../$src-en.tgz
        cd ..

        f=train.tags.$lang.$l
        tok=train.tags.$lang.tok.$l

        cat $orig/$lang/$f | \
        grep -v '<url>' | \
        grep -v '<talkid>' | \
        grep -v '<keywords>' | \
        sed -e 's/<title>//g' | \
        sed -e 's/<\/title>//g' | \
        sed -e 's/<description>//g' | \
        sed -e 's/<\/description>//g' | \
        perl $TOKENIZER -threads 8 -l $l > $tmp/$tok
        echo ""
    done
    perl $CLEAN -ratio 1.5 $tmp/train.tags.$lang.tok $src $tgt $tmp/train.tags.$lang.clean 1 175
    for l in $src $tgt; do
        mv $tmp/train.tags.$lang.clean.$l  $tmp/train.tags.$lang.$l
    done

    echo "pre-processing valid/test data..."
    for l in $src $tgt; do
        for o in `ls $orig/$lang/IWSLT14.TED*.$l.xml`; do
        fname=${o##*/}
        f=$tmp/${fname%.*}
        echo $o $f
        grep '<seg id' $o | \
            sed -e 's/<seg id="[0-9]*">\s*//g' | \
            sed -e 's/\s*<\/seg>\s*//g' | \
            sed -e "s/\’/\'/g" | \
        perl $TOKENIZER -threads 8 -l $l | > $f
        echo ""
        done
    done


    echo "creating train, valid, test..."
    for l in $src $tgt; do
        awk '{if (NR%23 == 0)  print $0; }' $tmp/train.tags.$src-$tgt.$l > $tmp/valid.en-$src.$l
        awk '{if (NR%23 != 0)  print $0; }' $tmp/train.tags.$src-$tgt.$l > $tmp/train.en-$src.$l

        cat $tmp/IWSLT14.TED.dev2010.$src-$tgt.$l \
            $tmp/IWSLT14.TEDX.dev2012.$src-$tgt.$l \
            $tmp/IWSLT14.TED.tst2010.$src-$tgt.$l \
            $tmp/IWSLT14.TED.tst2011.$src-$tgt.$l \
            $tmp/IWSLT14.TED.tst2012.$src-$tgt.$l \
            > $tmp/test.en-$src.$l
    done

    TRAIN=$tmp/train.all
    BPE_CODE=$prep/code
    rm -f $TRAIN
    for l in $src $tgt; do
        cat $tmp/train.en-$src.$l >> $TRAIN
    done
done
echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for src in ar de es fa he it nl pl; do
    for L in $src $tgt; do
        for f in train.en-$src.$L valid.en-$src.$L test.en-$src.$L; do
            echo "apply_bpe.py to ${f}..."
            python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $prep/$f
        done
    done
done
