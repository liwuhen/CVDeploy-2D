import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Run COCO mAP evaluation
# Reference: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
# import pdb; pdb.set_trace()
root_path= "/home/selflearning/opensource/HPC_Deploy/"
annotations_path = os.path.join(root_path, "install_nvidia/yolov5_bin/x86/workspace/gt_val.json")

root_path= "/home/selflearning/opensource/HPC_Deploy/"
results_file = os.path.join(root_path, "install_nvidia/yolov5_bin/x86/workspace/model_prediction.json")
cocoGt = COCO(annotation_file=annotations_path)
cocoDt = cocoGt.loadRes(results_file)
imgIds = sorted(cocoGt.getImgIds())
cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
cocoEval.params.imgIds = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
