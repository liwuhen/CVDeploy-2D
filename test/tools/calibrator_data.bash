#!/bin/bash

DATA_TYPE="coco"
SRC_PATH=""
DEST_PATH=""
NUM_IMAGES=0

if [ "${DATA_TYPE}" == "coco" ] ; then
    SRC_PATH="/home/selflearning/dataset/tinycoco/images/train2017"
    DEST_PATH="/home/selflearning/dataset/tinycoco/images/calib_data_1000"
    NUM_IMAGES=1000

elif [ "${DATA_TYPE}" == "voc" ] ; then
    SRC_PATH="/home/selflearning/dataset/VOC/images/train"
    DEST_PATH="/home/selflearning/dataset/VOC/images/calib_data_2000"
    NUM_IMAGES=2000
fi

echo "DATH_TYPE: $DATA_TYPE"
echo "SRC_PATH: $SRC_PATH"
echo "DEST_PATH: $DEST_PATH"
echo "NUM_IMAGES: $NUM_IMAGES"

python ./calibrator_data.py\
    --src_path "$SRC_PATH"\
    --dest_path "$DEST_PATH"\
    --num_images $NUM_IMAGES
