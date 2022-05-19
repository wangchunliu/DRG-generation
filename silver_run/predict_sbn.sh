#!/bin/bash

encodertype=$1
datatype=$2
DATASET=$3

GPU_ID=0
DATA_DIR="/data/p289796/SBN-data-silver/temp_data"
python ../data_preprocess/sbn_preprocess.py  -input_src ${DATA_DIR}/${DATASET}.txt -input_tgt ${DATA_DIR}/${DATASET}.txt.raw -text_type graph -if_anony normal
~/mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < ${DATA_DIR}/${DATASET}.txt.raw > ${DATA_DIR}/${DATASET}.txt.raw.tok -no-escape 
REF_FILE=${DATA_DIR}/${DATASET}.txt.graph.normal
TARG_FILE=${DATA_DIR}/${DATASET}.txt.raw.tok


MODELS_DIR="/data/p289796/SBN_data/gold-silver-deep/$datatype"
OUT_DIR=${MODELS_DIR}/preds/
MODEL=${MODELS_DIR}/best.pt

mkdir -p ${OUT_DIR}
for fname in ${MODELS_DIR}/$encodertype*; do
    f=${fname##*/}
    echo $f
    python ../translate.py \
	    -data_type $datatype \
        -model ${fname} \
        -src ${REF_FILE} \
        -tgt ${TARG_FILE} \
        -output ${OUT_DIR}/{$f}.pred.txt \
        -beam_size 5 \
        -batch_size 1 \
        -gpu ${GPU_ID} \
        -replace_unk \
        -max_length 500 > log.txt
done

~/mosesdecoder/scripts/tokenizer/detokenizer.perl -l en < ${OUT_DIR}/{$f}.pred.txt > ${OUT_DIR}/{$f}.detok.pred.txt
#python ../data_preprocess/postprocess_deAnony.py ${OUT_DIR}/{$f}.detok.pred.txt ${DATA_DIR}/${DATASET}.txt.alignment ${OUT_DIR}/{$f}.detok.pred.txt.dealign
python ~/project/e2e-metrics/measure_scores.py ${DATA_DIR}/${DATASET}.txt.raw ${OUT_DIR}/{$f}.detok.pred.txt

