#!/bin/bash


EVAL_DATA_TYPE="coco"
ROOT_PATH="/home/selflearning/opensource/HPC_Deploy"

if [ "${EVAL_DATA_TYPE}" == "coco" ] ; then
    GT_PATH=$ROOT_PATH/"install_nvidia/yolov5_bin/x86/workspace/gt_val.json"
    INFER_PATH=$ROOT_PATH/"install_nvidia/yolov5_bin/x86/workspace/model_prediction.json"

elif [ "${EVAL_DATA_TYPE}" == "voc" ] ; then
    GT_PATH=$ROOT_PATH/"install_nvidia/yolov5_bin/x86/workspace/gt_val.json"
    INFER_PATH=$ROOT_PATH/"install_nvidia/yolov5_bin/x86/workspace/model_prediction.json"
fi

echo "EVAL_DATA_TYPE: $EVAL_DATA_TYPE"
echo "GT_PATH: $GT_PATH"
echo "INFER_PATH: $INFER_PATH"

python ./test_benchmark.py\
    --gt_path "$GT_PATH"\
    --infer_path "$INFER_PATH"
