import os
import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Run COCO mAP evaluation
# Reference: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', type=str, help="gt label data path")
    parser.add_argument('--infer_path',type=str, help="infer result path")
    args = parser.parse_args()

    cocoGt = COCO(annotation_file=args.gt_path)
    cocoDt = cocoGt.loadRes(args.infer_path)
    imgIds = sorted(cocoGt.getImgIds())
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
