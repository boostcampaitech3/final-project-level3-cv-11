# from util import CropRoiImg
from util import Get_normal_bbox

from detection import mtcnn_detection, mtcnn_get_embeddings, mtcnn_recognition
from retinaface_utils.util import retinaface_detection

def Detection(img, args, model_args):
    # return bboxes position

    # ======== ML Part ============
    device = model_args['Device']

    if args['DETECTOR'] == 'mtcnn':
        bboxes = mtcnn_detection(model_args['Detection'], img, device)
    else:
        bboxes = retinaface_detection(model_args['Detection'], img, device)

    if args['DEBUG_MODE']:
        print(bboxes)
    # 이미지 범위 외로 나간 bbox값 범위 처리
    if bboxes is not None:
        bboxes = Get_normal_bbox(img.shape, bboxes)
    # =============================

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

    faces, unknown_embeddings = mtcnn_get_embeddings(model_args['Mtcnn'],
                                                     model_args['Recognition'],
                                                     img, bboxes, device)
    if args['DEBUG_MODE']:
        print(faces.shape)

    IoU_treshhold = 0.8
    iou_weights = ['unknown' for _ in bboxes]
    if known_ids is not None:
        if known_ids:
            # IoU 비교로 겹치는 대상을 known_ids.iou_weights에 저장
            known_ids.check_iou(bboxes)

    face_ids, result_probs = mtcnn_recognition(img, model_args['Face_db'],
                                            unknown_embeddings,
                                            args['RECOG_THRESHOLD'], known_ids)

    if args['DEBUG_MODE']:
        print(face_ids)
    # =============================

    return face_ids
