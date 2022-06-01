import cv2
import numpy as np


def GetFaceFeature(img):
    return []


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


def CropRoiImg(img, bboxes):
    roi_imgs = []
    for bbox in bboxes:
        # bbox: x, y, w, h
        y0 = bbox[1]
        y1 = bbox[1] + bbox[3]
        x0 = bbox[0]
        x1 = bbox[0] + bbox[2]

        roi_img = img[y0: y1, x0:x1]
        # 추가적으로 roi_img feature를 뽑아야 할지
        # GetFaceFeature(img)
        roi_imgs.append(roi_img)
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


def DrawRectImg(img, bboxes, face_ids):
    rect_color = (0, 0, 255) # BGR
    rect_thickness = 2 # 이미지 사이즈에 맞게 조절해야할지도
    font_scale = 1 # 위와 동일
    font_color = (0, 0, 255) # BGR
    font_thickness = 1 # 위와 동일
    
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
