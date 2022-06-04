import cv2
import numpy as np
from torch.nn.functional import interpolate
from torchvision.transforms import functional as F
from PIL import Image
import torch


def AddFaceData(_get_vector: bool, imgs: list=[]) -> list:
    assert type(imgs) == list, 'input list of image'

    # 기존 이미지를 활용하여 얼굴 데이터 확보
    assert imgs != [], 'img is empty'
    
    if _get_vector:
        # 벡터화하여 return
        for img in imgs:
            pass
        pass
    else:
        # 이미지 자체를 return
        return imgs


def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor


def imresample(img, sz):
    im_data = interpolate(img, size=sz, mode="area")
    return im_data


def crop_resize(img, box, image_size):
    if isinstance(img, np.ndarray):
        img = img[box[1]:box[3], box[0]:box[2]]
        out = cv2.resize(
            img,
            (image_size, image_size),
            interpolation=cv2.INTER_AREA
        ).copy()
    elif isinstance(img, torch.Tensor):
        img = img[box[1]:box[3], box[0]:box[2]]
        out = imresample(
            img.permute(2, 0, 1).unsqueeze(0).float(),
            (image_size, image_size)
        ).byte().squeeze(0).permute(1, 2, 0)
    else:
        out = img.crop(box).copy().resize((image_size, image_size), Image.BILINEAR)
    return out


def CropRoiImg(img, bboxes, image_size, save_path):
    # Crop image 저장 옵션 -> 중간에 unknown face의 데이터를 뽑아야할 때 씀
    # bbox에 맞춰 crop, output size 조절 옵션
    # batch mode 고려
    roi_imgs = []
    for bbox in bboxes:
        # bbox: x, y, w, h
        bbox = np.round(bbox).astype(int)
        face = crop_resize(img, bbox, image_size)
        face = F.to_tensor(np.float32(face))
        face = fixed_image_standardization(face)
        roi_imgs.append(face)

        if save_path is not None:
            #os.makedirs(os.path.dirname(save_path) + "/", exist_ok=True)
            #save_img(face, save_path)
            pass

    roi_imgs = torch.stack(roi_imgs)

    return roi_imgs


def Get_normal_bbox(size, bboxes):
    new_bboxes = None
    for bbox in bboxes:
        if bbox[0] < 0: bbox[0] = 0
        if bbox[1] < 0: bbox[1] = 0
        if bbox[2] > size[1]: bbox[2] = size[1]
        if bbox[3] > size[0]: bbox[3] = size[0]

        # 처리한 bbox의 상태가 이상하면 제거 처리
        if bbox[2] - bbox[0] > 0 or bbox[3] - bbox[1] > 0:
            bbox = np.expand_dims(bbox, 0)
            if new_bboxes is None:
                new_bboxes = bbox
            else:
                new_bboxes = np.concatenate([new_bboxes, bbox])
    return new_bboxes


def Mosaic(img, bboxes, face_ids, n):
    # filling NxN kernel's max or average value
    # img: original image
    # bboxes: mosaic target positions
    # n: kernel size

    for bbox, face_id in zip(bboxes, face_ids):
        if face_id == 'unknown':
            bbox = np.round(bbox).astype(int)
            # 대상이 너무 작아 모자이크가 안된다면 pass
            if bbox[2] - bbox[0] < n or bbox[3] - bbox[1] < n:
                continue
            roi = img[bbox[1]:bbox[3], bbox[0]:bbox[2]] 
           # 1/n 비율로 축소
            roi = cv2.resize(roi, ((bbox[2] - bbox[0])//n,
                                (bbox[3] - bbox[1])//n),
                                interpolation=cv2.INTER_AREA)
            # 원래 크기로 확대
            roi = cv2.resize(roi, ((bbox[2] - bbox[0]),
                                (bbox[3] - bbox[1])),
                                interpolation=cv2.INTER_NEAREST)
            img[bbox[1]:bbox[3], bbox[0]:bbox[2]] = roi

    return img


def DrawRectImg(img, bboxes, face_ids, known_ids):
    rect_color = (0, 0, 255) # BGR
    rect_thickness = 2 # 이미지 사이즈에 맞게 조절해야할지도
    font_scale = 1 # 위와 동일
    font_color = (0, 0, 255) # BGR
    font_thickness = 1 # 위와 동일
    
    cv2.putText(img, str(known_ids.cnt()), (10, 50), 1, 3, font_color, 3)
    for (bbox, face_id) in zip(bboxes, face_ids):
        if face_id != 'unknown':
            # bbox: x0, y0, x1, y1
            bbox = np.round(bbox).astype(int)
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                            rect_color, rect_thickness)
            cv2.putText(img, face_id, (bbox[0], bbox[1]-5),
                            1, font_scale, font_color, font_thickness)
    img_draw = img

    return img_draw


def CheckIoU(box1, box2):

    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(box1[0], box2[0]) # 좌
    y1 = max(box1[1], box2[1]) # 상
    x2 = min(box1[2], box2[2]) # 우
    y2 = min(box1[3], box2[3]) # 하

    # compute the width and height of the intersection
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)
    if w * h == 0:
        return 0.0
    
    # box = (x1, y1, x2, y2)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    inter = w * h
    iou = inter / (box1_area + box2_area - inter)

    return iou


class TrackingID():
    def __init__(self, IoU_thr, IoU_weight):
        self.known_ids = {}
        self.iou_weights = []
        self.iou_threshold = IoU_thr
        self.recog_weight = IoU_weight
        self.count = 0

    def get_ids(self, face_ids, bboxes):
        self.known_ids = {} # initialize
        for id, bbox in zip(face_ids, bboxes):
            if id != 'unknown':
                self.known_ids[id] = bbox

    def check_iou(self, bboxes):
        self.iou_weights = ['unknown' for _ in bboxes]
        for i, bbox in enumerate(bboxes):
            for name, known_bbox in self.known_ids.items():
                iou = CheckIoU(bbox, known_bbox)
                if iou > self.iou_threshold:
                    self.iou_weights[i] = name
    
    def __call__(self) -> dict:
        return self.known_ids

    def weights_list(self) -> list:
        return self.iou_weights

    def threshold(self) -> float:
        return self.iou_threshold
    
    def weight(self) -> float:
        return self.recog_weight

    def cnt(self):
        self.count += 1
        return self.count
