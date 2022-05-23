import cv2
import numpy as np
from math import ceil
from PIL import Image, ImageDraw
from facenet_pytorch import extract_face

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


def CropRoiImg(img, bboxes, threshold):
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


def Mosaic(img, bboxes, n, mode):
    # filling NxN kernel's max or average value
    # img: original image
    # bboxes: mosaic target positions
    # n: kernel size
    # mode: max = 0, average = 1
    # 아직 덜 짜서 커널 사이즈에 따라 bbox 영역을 벗어나기도 함

    if mode == 0: # max
        for bbox in bboxes:
            # bbox: x, y, w, h
            for col in range(ceil(bbox[3]/n)):
                for row in range(ceil(bbox[2]/n)):
                    max = np.max(img[bbox[1] + n*col:bbox[1] + n*(col+1),
                                     bbox[0] + n*row:bbox[0] + n*(row+1)])
                    img[bbox[1] + n*col:bbox[1] + n*(col+1),
                        bbox[0] + n*row:bbox[0] + n*(row+1)] = max
    elif mode == 1: # average
        for bbox in bboxes:
            # bbox: x, y, w, h
            for col in range(ceil(bbox[3]/n)):
                for row in range(ceil(bbox[2]/n)):
                    mean = np.mean(img[bbox[1] + n*col:bbox[1] + n*(col+1),
                                      bbox[0] + n*row:bbox[0] + n*(row+1)])
                    img[bbox[1] + n*col:bbox[1] + n*(col+1),
                        bbox[0] + n*row:bbox[0] + n*(row+1)] = mean
    return img


def DrawRectImg(img, bboxes, landmarks, face_ids=[]):
    rect_color = (0, 0, 255) # BGR
    rect_thickness = 2 # 이미지 사이즈에 맞게 조절해야할지도
    font_scale = 1 # 위와 동일
    font_color = (0, 0, 255) # BGR
    font_thickness = 2 # 위와 동일
        
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)
    for i, (box, point) in enumerate(zip(bboxes, landmarks)):
        draw.rectangle(box.tolist(), width=rect_thickness, outline=rect_color)
        for p in point : 
            # draw.circle((p-10).tolist() + (p+10).tolist(), outline='white')
            draw.ellipse((p-10).tolist() + (p+10).tolist(), outline='white') # DOT WANT
        extract_face(img, box, save_path='detected/detected_face_{}.png'.format(i))
        if face_ids:
            draw.text((int(box.tolist()[0]), int(box.tolist()[1])), face_ids[i])
    img_draw.save('annotated/annotated_faces.png')

    return img_draw