#!/usr/bin/env python3
"""
detect_end2end.py
纯 ONNXRuntime 推理 YOLOv5 end2end ONNX（已含 Efficient-NMS）
> python detect_end2end.py --weights best-nms-end2end.onnx --source zidane.jpg
"""
import argparse, time, cv2, numpy as np, onnxruntime as ort
from pathlib import Path
import mss
import onnxruntime as ort
print(ort.get_available_providers())


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    # 保持比例缩放 + 灰条
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    print(new_unpad)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    #dw, dh = np.mod(dw, 32), np.mod(dh, 32)
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    print(im.shape)
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # 框坐标映射回原图
    boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1 = x - w/2
    boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1 = y - h/2
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2 = x1 + w
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]  # y2 = y1 + h
    print("img1_shape:", img1_shape)
    print("img0_shape:", img0_shape)
    print("ratio_pad:", ratio_pad)
    if len(img0_shape) == 3:
        img0_shape = img0_shape[:2]
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0]
        pad = ratio_pad[1]
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes[:, :4] /= gain
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, img0_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, img0_shape[0])
    print(boxes)
    return boxes

def plot_boxes(im, boxes, color=(0, 255, 0)):
    #if colors is None:
        #colors = {i: tuple(map(int, cv2.applyColorMap(np.uint8([i*10]), cv2.COLORMAP_JET)[0,0])) for i in range(len(names))}
    for *xyxy, conf, cls in boxes:
        #label = f'{names[int(cls)]} {conf:.2f}'
        cv2.rectangle(im, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)
        #cv2.putText(im, label, (int(xyxy[0]), int(xyxy[1])-2), 0, 0.6, colors[int(cls)], 2)
    return im

class YOLOv5End2End:
    def __init__(self, weights, device='cpu'):
        #providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(weights, providers=providers)
        self.input_names = [x.name for x in self.session.get_inputs()]
        self.output_names = [x.name for x in self.session.get_outputs()]
        self.in_shape = self.session.get_inputs()[0].shape[-2:]  # [h, w]

    def __call__(self, img0):
        # 1. 前处理
        print(img0.shape)
        img, ratio, dwdh = letterbox(img0)
        print("==================")
        print(img.shape)
        print("==================")
        img = img[:, :, ::-1].transpose(2, 0, 1)  #  BGR → RGB HWC -> CHW
        img = np.ascontiguousarray(img[None], dtype=np.float32) / 255.0 #规一化的NCHW
        # 2. 推理
        t0 = time.time()
        outputs = self.session.run(None, {self.input_names[0]: img})[0]  # [n,6]
        print(f'ONNX {(time.time()-t0)*1000:.1f}ms, {len(outputs)} obj')
        print('output shape:', outputs.shape)
        print("==================")
        for row in outputs:
            print(row.shape)
        print("==================")
        outs = outputs[0]  # (25200, 26)

        outs = outputs[0]  # (25200, 26)

        boxes = []
        confidences = []
        class_ids = []
        filtered_boxes = []
        filtered_confidences = []
        filtered_class_ids = []
        for row in outs:
            # 解包前 4 个值
            x, y, w, h = row[:4]

            # 解包第 5 个值（置信度）
            conf = row[4]

            # 解包后 20 个值（类别分数）
            cls_scores = row[5:]

            max_cls_score = np.max(cls_scores)
            max_cls_id = np.argmax(cls_scores)

            if conf > 0.25 and max_cls_score > 0.25:
                boxes.append([x, y, w, h])
                confidences.append(float(conf * max_cls_score))
                class_ids.append(max_cls_id)

        iou_thres = 0.45  # NMS IOU threshold
        nms_indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.25, nms_threshold=iou_thres)
        print(nms_indices)
        print(nms_indices.shape)
        for i in nms_indices:
            box = boxes[i]
            filtered_boxes.append(box)
            conf = confidences[i]
            filtered_confidences.append(conf)
            cls_id = class_ids[i]
            filtered_class_ids.append(cls_id)
            print(f'Box: {box}  Conf: {conf:.3f}  Class: {cls_id}')
            # 格式化打印类别分数
        # 3. 坐标映射
        boxes1 = np.array(filtered_boxes)
        if len(nms_indices):
            boxes1 = scale_boxes(img.shape[2:], boxes1, img0.shape, (ratio, dwdh))
        filtered_confidences = np.array(filtered_confidences)[:, None]
        filtered_class_ids = np.array(filtered_class_ids)[:, None]
        combined_data = np.hstack((boxes1, filtered_confidences, filtered_class_ids))
        return combined_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='best.onnx', help='end2end onnx')
    parser.add_argument('--source', required=True, help='image/video')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--names', default='coco.names', help='class names, one per line')
    parser.add_argument('--project', default='runs/onnx', help='save results')
    opt = parser.parse_args()

    sct = mss.mss()
    monitor = {"left": 73, "top": 90, "width": 768, "height": 670}
    im0 = np.array(sct.grab(monitor))[:, :, :3] #BGR
    model = YOLOv5End2End(opt.weights)
    # 单张图 demo
    preds = model(im0)
    print(preds)
    im_out = plot_boxes(im0.copy(), preds)
    cv2.imshow('im_out', im_out)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()