#!/usr/bin/env python3
# 转换脚本 python export.py --weights best.pt --include engine --imgsz 640 --device 0 --half
# tensorrt 下载 python3 -m pip install tensorrt-cu12==10.4.* -f https://developer.download.nvidia.com/compute/redist/tensorrt/index.html
#0 images (1, 3, 640, 640) <class 'numpy.float16'> INPUT
#1 output0 (1, 25200, 26) <class 'numpy.float16'> OUTPUT
import argparse, time, cv2, numpy as np
from pathlib import Path
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import mss
import threading

JUMP_LONG_MS    = 0.35
import pyautogui
pyautogui.PAUSE = 0.0
flag_JUMP = False

def _hold_jump():
    global flag_JUMP
    """子线程里长按空格"""
    pyautogui.keyDown("space")
    time.sleep(JUMP_LONG_MS)
    pyautogui.keyUp("space")
    flag_JUMP = False

def send_key(action: str):
    global flag_JUMP
    if action == "FORWARD":
        pyautogui.keyDown("right")
    elif action == "FORWARD_JUMP":
        pyautogui.keyDown("right")
        if not flag_JUMP:
            flag_JUMP = True
            threading.Thread(target=_hold_jump, daemon=True).start()
    elif action == "RELEASE":
        pyautogui.keyUp("left")
        pyautogui.keyUp("right")


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw //= 2
    dh //= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

def scale_coords(img1_shape, boxes, img0_shape, ratio_pad=None):
    """将 [x1,y1,x2,y2] 映射回原图"""
    cx = boxes[:, 0].copy()
    cy = boxes[:, 1].copy()
    w = boxes[:, 2] - boxes[:, 0]
    boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1 = x - w/2
    boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1 = y - h/2
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2 = x1 + w
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]  # y2 = y1 + h
    boxes6 = np.concatenate([boxes, cx[:, None], cy[:, None]], axis=1)
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain, pad = ratio_pad
    boxes6[:, [0, 2]] -= pad[0]
    boxes6[:, [1, 3]] -= pad[1]
    boxes6[:, :6] /= gain
    boxes6[:, [0, 2, 4]] = boxes6[:, [0, 2, 4]].clip(0, img0_shape[1])
    boxes6[:, [1, 3, 5]] = boxes6[:, [1, 3, 5]].clip(0, img0_shape[0])
    return boxes6

def plot_boxes(im, preds, color=(0, 255, 0)):
    for row in preds:
        x1, y1, x2, y2, cx, cy, conf, cls = row.tolist()
        cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    """for pred in preds:
        print(type(pred), pred.shape, pred.dtype)
        for x1, y1, x2, y2, cx, cy, conf, cls in pred:
            cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        #cv2.putText(im, f'{int(cls)} {conf:.2f}', (int(xyxy[0]), int(xyxy[1]) - 5),
                    #cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)"""
    return im

class TRTEngineEnd2End:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.INFO)
        with open(engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # 假设 engine 只有 1 输入 1 输出
        self.inp_name = "images"  # YOLOv5 默认输入名
        self.out_name = "output0"  # YOLOv5 默认输出名
        self.inp_shape = tuple(self.engine.get_tensor_shape(self.inp_name))
        self.out_shape = tuple(self.engine.get_tensor_shape(self.out_name))

        # 分配 GPU/CPU 缓存
        self.d_in = cuda.mem_alloc(np.empty(self.inp_shape, dtype=np.float16).nbytes)
        self.d_out = cuda.mem_alloc(np.empty(self.out_shape, dtype=np.float16).nbytes)

    def __call__(self, img0):
        # 1. letterbox
        img, ratio, (dw, dh) = letterbox(img0, new_shape=self.inp_shape[-2:])
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
        img = np.ascontiguousarray(img[None], dtype=np.float16) / 255.0

        # 2. TensorRT 推理
        stream = cuda.Stream()
        cuda.memcpy_htod_async(self.d_in, img, stream)
        self.context.set_tensor_address(self.inp_name, int(self.d_in))
        self.context.set_tensor_address(self.out_name, int(self.d_out))
        #self.context.execute_async_v2(bindings=[int(self.d_in), int(self.d_out)], stream_handle=stream.handle)
        self.context.execute_async_v3(stream.handle)
        out = np.empty(self.out_shape, dtype=np.float16)
        cuda.memcpy_dtoh_async(out, self.d_out, stream)
        stream.synchronize()

        # 3. 后处理：out 形状 (keep_top_k, 7)
        #    有效检测数量 = 非零行
        out = out[0]
        box = out[:, :4]  # 4
        obj = out[:, 4:5]  # 1
        cls = out[:, 5:]  # 21
        conf = obj * cls
        score = conf.max(axis=1)
        cls_id = conf.argmax(axis=1)
        mask = score > 0.25
        box = box[mask]
        score = obj[mask]
        cls_id = cls_id[mask]
        box_list = box.tolist()  # [[x1,y1,x2,y2], ...]
        score_list = score.squeeze().tolist()  # [s1, s2, ...]
        iou_thres = 0.45  # NMS IOU threshold
        nms_indices = cv2.dnn.NMSBoxes(box_list, score_list, score_threshold=0.25, nms_threshold=iou_thres)
        box = box[nms_indices]
        score = score[nms_indices]
        cls_id = cls_id[nms_indices]
        boxes1 = scale_coords(img.shape[2:], box, img0.shape, (ratio, (dw, dh)))
        #preds = out[valid]
        #if len(preds) == 0:
        #    return np.empty((0, 6))
        # 坐标映射回原图
        #boxes = scale_coords(img.shape[2:], preds[:, :4], img0.shape, (ratio, (dw, dh)))
        return np.hstack([boxes1, score.reshape(-1, 1), cls_id.reshape(-1, 1)])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='best.engine', help='TensorRT end2end engine')
    opt = parser.parse_args()
    model = TRTEngineEnd2End(opt.weights)

    sct = mss.mss()
    mon = {"left": 73, "top": 90, "width": 768, "height": 670}
    while True:
        img0 = np.array(sct.grab(mon))[:, :, :3]
        preds = model(img0)
        im = plot_boxes(img0.copy(), preds)
        cv2.imshow('continuous', im)
        if cv2.waitKey(0) & 0xFF == 27:  # ESC 退出
            break
        mario = [(x1, y1, x2, y2, cx, cy, cls) for x1, y1, x2, y2, cx, cy, conf, cls in preds if cls in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)]
        enemy = [(x1, y1, x2, y2, cx, cy, cls) for x1, y1, x2, y2, cx, cy,conf, cls in preds if cls in (10, 14, 15)]
        cliff = [(x1, y1, x2, y2, cx, cy, cls) for x1, y1, x2, y2, cx, cy,conf, cls in preds if cls in (13, 19)]
        if not mario:
            continue
        if len(mario) > 1:
            continue
        if len(enemy) > 0:
            print("find enemy")
        x1, y1, x2, y2, cx, cy, cls = mario[0]
        mario_width = x2 - x1
        find_vaild_enemy = False
        for en in enemy:
            print(f"enemy type is :{en[6]}")
            dis = en[4] - cx
            if dis < mario_width * 3 and dis > 0:
                find_vaild_enemy = True #起跳
            if find_vaild_enemy:
                break
        find_vaild_cliff = False
        for c in cliff:
            dis = c[0] - x2
            print(dis)
            if dis < mario_width * 3 and dis > -10:
                find_vaild_cliff = True #起跳
            if find_vaild_cliff:
                break
        if find_vaild_enemy or find_vaild_cliff:
            send_key("FORWARD_JUMP")
            #print("FORWARD_JUMP")
        else:
            send_key("FORWARD")
            #print("FORWARD")




    #box = preds[:, :4].round(2)  # 坐标保留 2 位小数
    #conf = preds[:, 4:5].round(2)  # 置信度保留 2 位小数
    #cls_id = preds[:, 5].astype(int)  # 类别号转整数
    #im = plot_boxes(img0.copy(),preds)
    # 拼回去
    #cv2.imshow('trt-end2end', im)
    #cv2.waitKey(0)

if __name__ == '__main__':
    main()