#!/usr/bin/env bash

# Usage: test_fp16.sh pretrained_model_folder

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ "$#" -ne 1 ]; then
    echo "Usage: test_fp16.sh pretrained_model_folder"
    exit
fi

PRETRAINED_MODEL_PAHT=$1
CONFIG_FILE=$PRETRAINED_MODEL_PAHT/config.yaml
PRETRAINED_MODEL_FILE=$PRETRAINED_MODEL_PAHT/model_best.pth.tar

### Change accordingly
GPUS=0,1,2,3,4,5,6,7
NUM_GPUS=8
NUM_WORKERS=8
MASTER_PORT=1245

# ImageNet
DATA=$DIR/../datasets/ILSVRC2015/Data/CLS-LOC/

# test
CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port $MASTER_PORT \
    $DIR/../tools/main_fp16.py --cfg $CONFIG_FILE --workers $NUM_WORKERS \
    --fp16 \
    -p 100 --save-dir $PRETRAINED_MODEL_PAHT --pretrained --evaluate $DATA



