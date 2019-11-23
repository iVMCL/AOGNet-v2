#!/usr/bin/env bash

# Usage: train_fp16.sh config_filename

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ "$#" -ne 1 ]; then
    echo "Usage: train_fp16.sh relative_config_filename"
    exit
fi

CONFIG_FILE=$DIR/../$1

### Change accordingly
GPUS=0,1,2,3,4,5,6,7
NUM_GPUS=8
NUM_WORKERS=8
MASTER_PORT=1234

CONFIG_FILENAME="$(cut -d'/' -f2 <<<$1)"
CONFIG_BASE="${CONFIG_FILENAME%.*}"
SAVE_DIR=$DIR/../results/$CONFIG_BASE
mkdir -p $SAVE_DIR

# backup for reproducing results
cp $CONFIG_FILE $SAVE_DIR/config.yaml
cp -r $DIR/../models $SAVE_DIR
cp $DIR/../tools/main_fp16.py $SAVE_DIR

# ImageNet
DATA=$DIR/../datasets/ILSVRC2015/Data/CLS-LOC/

# train
CUDA_VISIBLE_DEVICES=$GPUS python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port $MASTER_PORT \
    $DIR/../tools/main_fp16.py --cfg $CONFIG_FILE --workers $NUM_WORKERS \
    --fp16 --static-loss-scale 128 \
    -p 100 --save-dir $SAVE_DIR $DATA  \
    2>&1 | tee $SAVE_DIR/log.txt

