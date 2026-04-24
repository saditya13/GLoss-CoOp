#!/bin/bash

# Usage: ./run_all_losses.sh CFG CTP NCTX SHOTS CSC
CFG=$1
CTP=$2
NCTX=$3
SHOTS=$4
CSC=$5

if [ -z "$CFG" ]; then
    echo "Usage: $0 CFG CTP NCTX SHOTS CSC"
    echo "Example: $0 vit_b16 end 16 16 False"
    exit 1
fi

DATA=/home/aditya/CoOp/Data
TRAINER=CoOp
SIGMA=4.5
DATASETS=(caltech101 dtd eurosat fgvc_aircraft food101 oxford_flowers oxford_pets ucf101)  # <-- your 4 datasets here

echo "=========================================="
echo "CFG=${CFG}, CTP=${CTP}, NCTX=${NCTX}, SHOTS=${SHOTS}, CSC=${CSC}"
echo "SIGMA=${SIGMA}"
echo "DATASETS=${DATASETS[@]}"
echo "=========================================="


for DATASET in "${DATASETS[@]}"
do
    echo "########## DATASET: ${DATASET} ##########" 

    echo "--- Running cross_entropy ---"
    bash ./scripts/coop/main.sh ${DATASET} ${CFG} ${CTP} ${NCTX} ${SHOTS} ${CSC} cross_entropy

    echo "--- Running gloss (sigma=${SIGMA}) ---"
    bash ./scripts/coop/main.sh ${DATASET} ${CFG} ${CTP} ${NCTX} ${SHOTS} ${CSC} gloss ${SIGMA}
done