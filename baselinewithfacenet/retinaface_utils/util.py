import torch
import cv2
import numpy as np

from retinaface_utils.layers.functions.prior_box import PriorBox
from retinaface_utils.utils.box_utils import decode, decode_landm
from retinaface_utils.utils.nms.py_cpu_nms import py_cpu_nms
from retinaface_utils.data.config import cfg_mnet

def retinaface_detection(model, img_raw, device):
    img, scale, resize = retinaface_preprocess(img_raw, device)
    loc, conf, landms = model(img)  # forward pass
    n_dets = retinaface_postprocess(loc, conf, landms, scale, resize, img.shape, device)
    return n_dets

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
    return dets[:, :4]
