import torch
from util import Get_normal_bbox

from detection import get_embeddings, recognizer, deepsort_recognizer
from retinaface_utils.util import retinaface_detection
from detect_face import yolo_detection
from util import xyxy2xywh


def Detection(img, args, model_args):
    '''
        return bboxes position
    '''
    device = model_args['Device']

    if args['DETECTOR'] == 'retinaface':
        bboxes, probs = retinaface_detection(model_args['Detection'], img, device)
    else: # yolo
        bboxes, probs = yolo_detection(model_args['Detection'], img, device)

    # 이미지 범위 외로 나간 bbox값 범위 처리
    if bboxes is not None:
        bboxes = Get_normal_bbox(img.shape, bboxes)

    if args['DEBUG_MODE']:
        print(bboxes)

    return bboxes, probs


def Recognition(img, bboxes, args, model_args, id_name=None, identities=None):
    '''
    return unknown face bboxes and ID: dictionary type, name: id's bbox

    특정인으로 분류가 안된 이미지들은 놔두고, 특정인으로 분류된 bbox는 0로 값을 바꾸고
    recog_bboxes에 Name(key): bbox(Value)로 넣어둠.
    dectection된 얼굴과 특정 대상 얼굴과 비교.
    '''
    device = model_args['Device']

    faces, unknown_embeddings = get_embeddings(model_args['Recognition'],
                                                     img, bboxes, device)                  
    if args['PROCESS_TARGET'] == 'Image' or not args['TRACKING']:
        face_ids, result_probs = recognizer(model_args['Face_db'],
                                        unknown_embeddings,
                                        args['RECOG_THRESHOLD'])
    else: 
        face_ids, result_probs = deepsort_recognizer(model_args['Face_db'],
                                                    unknown_embeddings,
                                                    args['RECOG_THRESHOLD'],
                                                    id_name, 
                                                    identities)
    if args['DEBUG_MODE']:
        print(face_ids)
        print(result_probs)

    return face_ids, result_probs


def Deepsort(img, bboxes, probs, deepsort):
    if bboxes is not None and len(bboxes):
        
        # bboxes = bboxes * scale      
        bbox_xywh = xyxy2xywh(bboxes)   
        # bbox_xywh[:, 2:] = bbox_xywh[:, 2:] * (1 + margin_ratio)

        outputs = deepsort.update(bbox_xywh, probs, img) # (#ID, 5) x1,y1,x2,y2,track_ID
    
    else:
        outputs = torch.zeros((0, 5))

    return outputs