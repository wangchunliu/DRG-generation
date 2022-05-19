#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=15:00:00
#SBATCH --mem=100GB
encodertype=$1
datatype=$2

GPUIDS=0
DATA_DIR=/data/p289796/SBN-data/gold-silver-deep/$datatype
OUT_DIR=/data/p289796/SBN_data/gold-silver-deep/$datatype
BATCH_SIZE=32
NUM_LAYERS=1

mkdir -p ${OUT_DIR}

python ../train.py \
    -activation 'relu' \
    -highway 'tanh' \
    -n_gcn_layer 1 \
    -gcn_edge_dropout 0.1 \
    -gcn_dropout 0.1 \
    -data_type $datatype \
    -data ${DATA_DIR}/sbn \
    -save_model ${OUT_DIR}/$encodertype \
    -layers ${NUM_LAYERS} \
    -report_every 1000 \
    -train_steps 60001 \
    -valid_steps 2000 \
    -rnn_size 750 \
    -word_vec_size 750 \
    -gcn_vec_size 750 \
    -decoder_type rnn \
    -batch_size ${BATCH_SIZE} \
    -max_generator_batches 50 \
    -save_checkpoint_steps 2000 \
    -decay_steps 1000 \
    -optim sgd \
    -max_grad_norm 2 \
    -learning_rate_decay 0.95 \
    -start_decay_steps 10000 \
    -learning_rate 1 \
    -dropout 0.5 \
    -gpu_ranks ${GPUIDS} \
    -seed 3333\
    -keep_checkpoint 1\
    -copy_attn \
    -encoder_type $encodertype

   
  
