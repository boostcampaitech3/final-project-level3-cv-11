import argparse
import copy
import cv2
import numpy as np
import time
import torch

from face_detection.retinaface.modules.prior_box import PriorBox
from face_detection.retinaface.modules.box_utils import decode, decode_landm
from face_detection.retinaface.modules.py_cpu_nms import py_cpu_nms
from configs.model.retinaface_config import cfg_mnet

from face_detection.yolov5_face.yolo_utils.datasets import letterbox
from face_detection.yolov5_face.yolo_utils.general import check_img_size, non_max_suppression_face, scale_coords
from face_detection.yolov5_face.yolo_utils.torch_utils import time_synchronized


# ----- MTCNN ----- #

def mtcnn_detection(model, img, device):
    bboxes, probs = model.detect(img, landmarks=False)
    return bboxes


# ----- RetinaFace ----- #

def retinaface_detection(model, img_raw, device):
    img, scale, resize = retinaface_preprocess(img_raw, device)
    loc, conf, landms = model(img)  # forward pass
    n_dets, probs = retinaface_postprocess(loc, conf, landms, scale, resize, img.shape, device)
    return n_dets, probs

def retinaface_preprocess(img_raw, device):
    img = np.float32(img_raw)
    target_size = 1600
    max_size = 2150
    im_shape = img.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    resize = float(target_size) / float(im_size_min)

    # prevent bigger axis from being more than max_size:
    if np.round(resize * im_size_max) > max_size:
        resize = float(max_size) / float(im_size_max)
    
    if True: # 원본 데이터 그대로 유지
        resize = 1
    if resize != 1:
        img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123) # Normalize channel
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0) # B, C, H, W
    img = img.to(device)
    scale = scale.to(device)
    
    return img, scale, resize

def retinaface_postprocess(loc, conf, landms, scale, resize, img_shape, device):
    im_height, im_width = img_shape[-2:]
    
    ########################
    ### Box post process ###
    ########################
    priorbox = PriorBox(cfg_mnet, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg_mnet['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg_mnet['variance'])
    scale1 = torch.Tensor([img_shape[3], img_shape[2], img_shape[3], img_shape[2],
                           img_shape[3], img_shape[2], img_shape[3], img_shape[2],
                           img_shape[3], img_shape[2]])
    
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()
    
    # ignore low scores
    inds = np.where(scores > 0.5)[0] # confidence_threshold 0.02
    if len(inds) ==0 :
        return None
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]
    
    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, 0.4) # nms_threshold 0.4
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    # dets = dets[:args.keep_top_k, :]
    # landms = landms[:args.keep_top_k, :]
    
    # dets = np.concatenate((dets, landms), axis=1)
    return dets[:, :4], dets[:, 4]


# ----- YOLOv5 ----- #

def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    #clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords

def yolo_detection2(model, image, device):
    # Load model
    img_size = 768 # 조정 가능.
    conf_thres = 0.3
    iou_thres = 0.5

    img0 = copy.deepcopy(image)
    h0, w0 = image.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

    img = letterbox(img0, new_shape=imgsz)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

    # Run inference
    t0 = time.time()

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)

    # print('img.shape: ', img.shape)
    # print('ori image.shape: ', image.shape)
    bboxes = []
    probs = []
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], image.shape).round()

            for j in range(det.size()[0]):
                xyxy = det[j, :4].view(-1).tolist()
                bboxes.append(xyxy)
                prob = det[j, 4].view(-1).tolist()
                probs.append(prob)
    # print(bboxes)
    return np.array(bboxes), np.array(probs)

def yolo_detection(model, image, device):
    # Load model
    img_size = 768 # 조정 가능.
    conf_thres = 0.3
    iou_thres = 0.5

    img0 = copy.deepcopy(image)
    h0, w0 = image.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

    img = letterbox(img0, new_shape=imgsz)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

    # Run inference
    t0 = time.time()

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)

    # print('img.shape: ', img.shape)
    # print('ori image.shape: ', image.shape)
    bboxes = []
    probs = []
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], image.shape).round()

            for j in range(det.size()[0]):
                xyxy = det[j, :4].view(-1).tolist()
                bboxes.append(xyxy)
                prob = det[j, 4].view(-1).tolist()
                probs.append(prob)
    # print(bboxes)
    return np.array(bboxes), np.array(probs)
