import cv2
import numpy as np
from torch.nn.functional import interpolate
from torchvision.transforms import functional as F
from PIL import Image
import torch


def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor


def imresample(img, sz):
    im_data = interpolate(img, size=sz, mode="area")
    return 


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


def Mosaic(img, bboxes, identities, n, id_name=None):
    # filling NxN kernel's max or average value
    # img: original image
    # bboxes: mosaic target positions
    # n: kernel size

    for bbox, face_id in zip(bboxes, identities):
        if id_name:
            face_id = id_name[face_id]

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


def DrawRectImg(img, bboxes, identities, id_name=None):
    rect_color = (0, 0, 255) # BGR
    rect_thickness = 2 # 이미지 사이즈에 맞게 조절해야할지도
    font_scale = 1 # 위와 동일
    font_color = (0, 0, 255) # BGR
    font_thickness = 1 # 위와 동일
    
    for bbox, face_id in zip(bboxes, identities):
        if id_name:
            face_id = id_name[face_id]
        if face_id != 'unknown':
            # bbox: x0, y0, x1, y1
            bbox = np.round(bbox).astype(int)
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                            rect_color, rect_thickness)
            cv2.putText(img, face_id, (bbox[0], bbox[1]-5),
                            1, font_scale, font_color, font_thickness)

    return img


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y