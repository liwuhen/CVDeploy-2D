import cv2
import argparse
import numpy as np
import time
import torch
import csv
import os
import onnxruntime as ort
from torchvision.ops import batched_nms
from typing import Dict, Tuple, Sequence

class_name = ["aeroplane","bicycle","bird","boat","bottle",
              "bus","car","cat","chair","cow",
              "diningtable","dog","horse","motorbike","person",
              "pottedplant","sheep","sofa","train","tvmonitor"]

class YoloProject():
    def __init__(self, modelpath,
                 confThreshold=0.5, nmsThreshold=0.2, objThreshold=0.5):
        self.classes = class_name
        self.objThreshold  = objThreshold
        self.confThreshold = confThreshold
        self.nmsThreshold  = nmsThreshold
        self.input_hight   = 640
        self.input_width   = 640
        self.num_classes   = len(self.classes)

        self.mean = np.array([123.675, 116.28, 103.53]).reshape((3, 1, 1)).astype(np.float32)
        self.std  = np.array([58.395, 57.12, 57.375]).reshape((3, 1, 1)).astype(np.float32)

        self.net = ort.InferenceSession(modelpath, providers = ['DmlExecutionProvider'])

    def resize_image(self, im, new_shape: Tuple[int, int] = (640, 640), color=(114, 114, 114),
                     auto=False, scaleFill=False, scaleup=False, stride=32):

        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        # import pdb; pdb.set_trace()
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im

    def preprocess(self, img, img_id):
        # import pdb; pdb.set_trace()
        img = self.resize_image(img)
        img = img.transpose((2, 0, 1))        # HWC to CHW
        img = img[::-1, :, :]                 # BGR to RGB
        img = np.ascontiguousarray(img)       # contiguous
        img = img.astype(np.float32)
        img = img / 255.0
        with open("/home/selflearning/opensource/HPC_Deploy/modules/config/model/yolov5/" + str(img_id) +"_preprocess.data", "wb") as f:
            f.write(img.tobytes())
        img = img[None]                       # expand for batch dim
        return img

    def preprocess_bin(self, img_id):
        # 读取二进制文件
        bin_path = '/home/selflearning/opensource/HPC_Deploy/install_nvidia/prepross.bin'
        preprocessed_data = np.fromfile(bin_path, dtype=np.float32)
        # 重塑数组形状为 CHW 格式 (3, 470, 460)
        preprocessed_data = preprocessed_data.reshape(3, 640, 640)
        preprocessed_data = preprocessed_data[None]

        return preprocessed_data

    def test_preprocess(self, img, img_id):
        # 验证前处理的结果
        # 读取二进制文件
        bin_path = '/home/selflearning/opensource/HPC_Deploy/install_nvidia/prepross.bin'
        preprocessed_data = np.fromfile(bin_path, dtype=np.float32)

        # 重塑数组形状为 CHW 格式 (3, 470, 460)
        preprocessed_data = preprocessed_data.reshape(3, 640, 640)

        # 对比当前预处理结果
        # import pdb; pdb.set_trace()
        current_preprocess = self.preprocess(img, img_id)

        # 计算差异
        diff = np.abs(preprocessed_data - current_preprocess[0])
        max_diff  = np.max(diff)
        mean_diff = np.mean(diff)
        mse = np.mean(np.square(diff))  # 计算均方误差

        print(f"Max difference: {max_diff}")
        print(f"Mean difference: {mean_diff}")
        print(f"MSE: {mse}")

        return

    def nms(self, detectbox, iou_thres=0.2):
        '''
        :param boxes: decode 解码排序后的 boxes [n,6] 6 = x1,y1,x2,y2,conf,label
        :param iou_thresh: iou 阈值
        :return: 经过 NMS 的 boxes
        '''
        # 利用 remove_flags 标记需要去除的 box
        remove_flags = [False] * len(detectbox)
        # 保留下的 box
        keep_boxes = []
        for i in range(len(detectbox)):
            if remove_flags[i]:
                continue
            ibox = detectbox[i]
            keep_boxes.append(ibox)
            for j in range(len(detectbox)):
                if remove_flags[j]:
                    continue
                jbox = detectbox[j]
                # 只有同一张图片中的同一个类别的 box 才计算 iou
                if ibox[5] != jbox[5]:
                    continue
                # 计算 iou，若大于阈值则标记去除
                if self.boxes_iou(ibox, jbox) > iou_thres:
                    remove_flags[j] = True

        return np.array(keep_boxes)

    def xywh2xyxy(self, x):
        '''
        :param x: decode 解码排序后的 boxes [n,6] 6 = cx,cy,width,height,objconf,classconf (feature map level)
        :return: left,top,right,bottom,objconf,classconf
        '''
        y = np.copy(x)

        y[0, ...] = x[0, ...] - x[2, ...] * 0.5  # top left x
        y[1, ...] = x[1, ...] - x[3, ...] * 0.5  # top left y
        y[2, ...] = x[0, ...] + x[2, ...] * 0.5  # bottom right x
        y[3, ...] = x[1, ...] + x[3, ...] * 0.5  # bottom right y

        return y

    def boxes_iou(self, box1, box2):
        def box_area(box):
            return (box[2] - box[0]) * (box[3] - box[1])

        left   = max(box1[0], box2[0])
        top    = max(box1[1], box2[1])
        right  = min(box1[2], box2[2])
        bottom = min(box1[3], box2[3])
        cross  = max(0, right-left) * max(0, bottom-top)
        union  = box_area(box1) + box_area(box2) - cross
        if cross == 0 or union == 0:
            return 0
        return cross / union

    def clip_boxes(self, boxes, shape):
        '''
        :param boxes: decode 解码排序后的 boxes [n,6] 6 = x1,y1,x2,y2,conf,label (origin image level)
        :param shape: origin image shape
        :return: origin image detection box(origin image level)
        '''
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2

    def scale_boxes(self, dstshape, boxes, srcshape, ratio_pad=None):
        '''
        :param dstshape: dst image shape
        :param boxes: decode 解码排序后的 boxes [n,6] 6 = x1,y1,x2,y2,conf,label (feature map level)
        :param srcshape: src image shape
        :return: origin image detection box(origin image level)
        '''
        if ratio_pad is None:  # calculate from img0_shape
            # import pdb;pdb.set_trace()
            gain = min(dstshape[0] / srcshape[0], dstshape[1] / srcshape[1])  # gain  = old / new
            gain = min(gain, 1.0)
            pad  = ((dstshape[1] - srcshape[1] * gain) * 0.5, (dstshape[0] - srcshape[0] * gain)* 0.5)  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad  = ratio_pad[1]

        boxes[..., [0, 2]] -= pad[0]  # x padding
        boxes[..., [1, 3]] -= pad[1]  # y padding
        boxes[..., :4] /= gain
        # self.clip_boxes(boxes, srcshape)
        return boxes

    def postprocess(self, frame, outs):
        boxes = []
        for detection in outs:  # cx, cy, width, height, cls, kpts
            if detection[4] > self.objThreshold:
                objconf  = detection[4]
                classId  = detection[5:].argmax()
                class_pro= detection[5+classId]
                confidence = objconf * class_pro
                if confidence > self.confThreshold:
                    detection = self.xywh2xyxy(detection)

                    boxes.append([detection[0], detection[1], detection[2], detection[3], confidence, classId])

        # 输入图像分辨率上的模型预测框
        boxes = np.array(boxes)
        print("decode boxes: ", boxes.shape)
        # 将 boxes 按照置信度高低排序
        boxes = sorted(boxes.tolist(), key= lambda x : x[4], reverse=True)
        boxes = self.nms(boxes, iou_thres=self.nmsThreshold)

        print("nms boxes: ", boxes.shape)

        # 输入图像分辨率上的模型预测框 ==> 映射回原图上尺寸
        # import pdb; pdb.set_trace()
        if ( len(boxes) != 0 ):
            boxesss = self.scale_boxes((self.input_hight, self.input_width), boxes, frame.shape[:2])
        else:
            boxesss = []

        for box in boxesss:
            frame = self.drawPred(frame, int(box[5]), box[4], box[0], box[1], box[2], box[3])
        return frame

    def drawPred(self, frame, classId, conf, left, top, right, bottom):

        # Draw a bounding box.
        cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=1)

        label = '%.2f' % conf
        label = '%s %s' % (self.classes[classId], label)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv2.rectangle(frame, (int(left), int(top) - int(round(1 * labelSize[1]))), (int(left) + int(round(0.8 * labelSize[0])),
                                                                                 int(top) ), (255,0,0), cv2.FILLED)
        cv2.putText(frame, label, (int(left), int(top-2)), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        return frame

    def detect(self, srcimg, img_id):
        # img  = self.preprocess_bin(img_id)
        img  = self.preprocess(srcimg, img_id)
        outs = self.net.run(None, {self.net.get_inputs()[0].name: img})
        srcimg = self.postprocess(srcimg, outs[0].squeeze(axis=0))
        return srcimg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath',   type=str, default='/home/selflearning/dataset/VOC/images/val', help="image path")
    parser.add_argument('--modelpath', type=str, default='/home/selflearning/opensource/HPC_Deploy/install_nvidia/yolov5_bin/x86/config/model/best.onnx')
    parser.add_argument('--confThreshold', default=0.25, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold',  default=0.65, type=float, help='nms iou thresh')
    parser.add_argument('--objThreshold',  default=0.25, type=float, help='object confidence')
    args = parser.parse_args()

    atnet = YoloProject(args.modelpath,
                        confThreshold=args.confThreshold,
                        nmsThreshold=args.nmsThreshold,
                        objThreshold=args.objThreshold)
    #推理图片
    for img in sorted(os.listdir(args.imgpath)):
        srcimg = cv2.imread(os.path.join(args.imgpath, img))
        img_id = int(os.path.splitext(os.path.basename(os.path.join(args.imgpath, img)))[0].lstrip('0'))
        srcimg = atnet.detect(srcimg, img_id)

        cv2.imshow("View", srcimg)
        cv2.waitKey(0)
