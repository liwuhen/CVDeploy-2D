#!/bin/bash

DATA_TYPE="voc"
DATA_PATH=""
LABEL_PATH=""
SAVE_PATH=""

if [ "${DATA_TYPE}" == "coco" ] ; then
    DATA_PATH="/home/selflearning/dataset/tinycoco/images/val2017"
    LABEL_PATH="/home/selflearning/dataset/tinycoco/labels/val2017"
    SAVE_PATH="/home/selflearning/opensource/HPC_Deploy/install_nvidia/yolov5_bin/x86/workspace/gt_val.json"

elif [ "${DATA_TYPE}" == "voc" ] ; then
    DATA_PATH="/home/selflearning/dataset/VOC/images/val"
    LABEL_PATH="/home/selflearning/dataset/VOC/labels/val"
    SAVE_PATH="/home/selflearning/opensource/HPC_Deploy/install_nvidia/yolov5_bin/x86/workspace/gt_val.json"
fi

echo "DATH_TYPE: $DATA_TYPE"
echo "DATA_PATH: $DATA_PATH"
echo "LABEL_PATH: $LABEL_PATH"
echo "SAVE_PATH: $SAVE_PATH"

python ./yolotojson.py --data_type "$DATA_TYPE"\
    --data_path "$DATA_PATH"\
    --label_path "$LABEL_PATH"\
    --save_path "$SAVE_PATH"
