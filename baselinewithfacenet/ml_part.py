from util import Get_normal_bbox
import numpy as np
import cv2

from detection import get_embeddings, recognizer
from retinaface_utils.util import retinaface_detection

def Detection(img, args, model_args):
    # return bboxes position

    # ======== ML Part ============
    device = model_args['Device']

    bboxes = retinaface_detection(model_args['Detection'],img, device)

    # 이미지 범위 외로 나간 bbox값 범위 처리
    if bboxes is not None:
        bboxes = Get_normal_bbox(img.shape, bboxes)
    # =============================

    if args['DEBUG_MODE']:
        print(bboxes)

    return bboxes


def Recognition(img, bboxes, args, model_args, known_ids):
    # return unknown face bboxes and ID: dictionary type, name: id's bbox
    # 특정인으로 분류가 안된 이미지들은 놔두고, 특정인으로 분류된 bbox는 0로 값을 바꾸고
    # recog_bboxes에 Name(key): bbox(Value)로 넣어둠

    #roi_imgs = CropRoiImg(img, bboxes)
    # dectection된 얼굴과 특정 대상 얼굴과 비교

    # known_ids: {Name: bbox, ...}
    # ======== ML Part ============
    device = model_args['Device']

    faces, unknown_embeddings = get_embeddings(model_args['Recognition'],
                                            img, bboxes, device)
                                            
    if args['DEBUG_MODE']:
        print(faces.shape)
        print(unknown_embeddings.shape)

        if known_ids:
            # IoU 비교로 겹치는 대상을 known_ids.iou_weights에 저장
            known_ids.check_iou(bboxes)
            # 임시 이전 프레임 표시
            for name, known_bbox in known_ids.known_ids.items():
                known_bbox = np.round(known_bbox).astype(int)
                cv2.putText(img, name, (known_bbox[0], known_bbox[1]-5),
                                1, 1, (255, 0, 0), 1)
                cv2.rectangle(img, (known_bbox[0], known_bbox[1]), (known_bbox[2], known_bbox[3]),
                                (255, 0, 0), 2)

    face_ids, result_probs = recognizer(model_args['Face_db'],
                                        unknown_embeddings,
                                        args['RECOG_THRESHOLD'], known_ids)

    if args['DEBUG_MODE']:
        print(result_probs)
    # =============================

    return face_ids
