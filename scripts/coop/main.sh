#!/bin/bash

# custom config    
DATA=/home/aditya/CoOp/Data  #/home/aditya/tcacode/TCA 
time=$(date +"%Y-%m-%d %T")
TRAINER=CoOp

DATASET=$1
CFG=$2  # config file
CTP=$3  # class token position (end or middle)
NCTX=$4  # number of context tokens
SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
CSC=$6  # class-specific context (False or True)
loss=$7  # loss type for CoOp trainer: cross_entropy or gloss
for SEED in 1 #2 3 
do
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}/${loss}_${TIMESTAMP}
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --coop-loss-type ${loss} \
    TRAINER.COOP.N_CTX ${NCTX} \
    TRAINER.COOP.CSC ${CSC} \
    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
    DATASET.NUM_SHOTS ${SHOTS}
done