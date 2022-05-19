#!/bin/bash

datatype=$1

DATA_DIR="/data/p289796/SBN-data-silver/temp_data"
mkdir -p ${DATA_DIR}
cp -r /data/p289796/corpus/en-sbn/gold-silver/*  ${DATA_DIR}/
python ../data_preprocess/sbn_preprocess.py  -input_src ${DATA_DIR}/train.txt -input_tgt ${DATA_DIR}/train.txt.raw -text_type graph -if_anony normal 
python ../data_preprocess/sbn_preprocess.py  -input_src ${DATA_DIR}/dev.txt -input_tgt ${DATA_DIR}/dev.txt.raw -text_type graph -if_anony normal
python ../data_preprocess/sbn_preprocess.py  -input_src ${DATA_DIR}/test.txt -input_tgt ${DATA_DIR}/test.txt.raw -text_type graph -if_anony normal
~/mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < ${DATA_DIR}/train.txt.raw > ${DATA_DIR}/train.txt.raw.tok -no-escape 
~/mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < ${DATA_DIR}/dev.txt.raw > ${DATA_DIR}/dev.txt.raw.tok -no-escape 
~/mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < ${DATA_DIR}/test.txt.raw > ${DATA_DIR}/test.txt.raw.tok -no-escape 


OUT_DIR=/data/p289796/SBN-data/gold-silver-deep/$datatype
mkdir -p ${OUT_DIR}


python ../preprocess.py \
    -seed 123 \
    -reentrancies \
    -data_type $datatype \
    -train_src ${DATA_DIR}/train.txt.graph.normal \
    -train_tgt ${DATA_DIR}/train.txt.raw.tok \
    -valid_src ${DATA_DIR}/dev.txt.graph.normal \
    -valid_tgt ${DATA_DIR}/dev.txt.raw.tok \
    -save_data ${OUT_DIR}/sbn \
    -src_words_min_frequency 1 \
    -tgt_words_min_frequency 1 \
    -src_seq_length 500 \
    -tgt_seq_length 500 \
    -dynamic_dict

