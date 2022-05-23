from util import CropRoiImg


def Detection(img, threshold):
    # return bboxes position

    # ======== ML Part ============
        
    # =============================

    return []


def Recognition(img, bboxes, threshold):
    # return unknown face bboxes and ID: dictionary type, name: id's bbox
    # 특정인으로 분류가 안된 이미지들은 놔두고, 특정인으로 분류된 bbox는 0로 값을 바꾸고
    # recog_bboxes에 Name(key): bbox(Value)로 넣어둠

    roi_imgs = CropRoiImg(img, bboxes)
    # dectection된 얼굴과 특정 대상 얼굴과 비교

    # ======== ML Part ============
    for img in roi_imgs:
        pass
    # =============================

    recog_bboxes = {}
    return bboxes, recog_bboxes
