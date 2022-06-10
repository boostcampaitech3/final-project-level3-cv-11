import os
import pickle

from web.fastapi_engine.database import load_face_db
from web.fastapi_engine import ml_part as ML
from web.fastapi_engine.util import Mosaic, DrawRectImg


def init_model_args(args, model_detection=None, model_recognition=None, algo_tracking=None):
    model_args = {}
    # 초기에 불러올 모델을 설정하는 공간입니다.
    device = args["DEVICE"]
    model_args['Device'] = args["DEVICE"]
    if args['DEBUG_MODE']:
        print('Running on device : {}'.format(device))
    
    if args["DO_DETECTION"]:
        # 1. Load Detection Model
        model_args["Detection"] = model_detection
        
        if args["DO_RECOGNITION"]:
            # 2. Load Recognition Models
            model_args["Recognition"] = model_recognition
            
            # Load Face DB
            model_args['Face_db_path'] = f".database/{args['USERNAME']}/{args['WHICH_DETECTOR']}"
            model_args['Face_db'] = load_face_db(model_args['Face_db_path'])
            
            if args["DO_TRACKING"]:
                # 3. Load Tracking Algorithm
                model_args["Tracking"] = algo_tracking
    
    return model_args


def SaveSingleEmbedding(img, args, model_args):
    process_target = args['PROCESS_TARGET']

    # Object Detection
    bboxes, probs = ML.Detection(img, args, model_args)
    if bboxes is None: return img

    # Object Recognition
    _, _, face_embeddings = ML.Recognition(img, bboxes, args, model_args)
    
    faces_img, embedding_data = face_embeddings.get_data()
    
    embedding_data = embedding_data[0].numpy() # save only first embedding
    if args["SAVE_FACE_NAME"] in model_args['Face_db']:
        model_args['Face_db'][args["SAVE_FACE_NAME"]].append(embedding_data)
    else:
        model_args['Face_db'][args["SAVE_FACE_NAME"]] = [embedding_data]

    with open(os.path.join(model_args['Face_db_path'], "face_db"), "wb") as f:
        pickle.dump(model_args['Face_db'], f)

    return 


def ProcessImage(img, args, model_args):
    process_target = args['PROCESS_TARGET']

    # Object Detection
    bboxes, probs = ML.Detection(img, args, model_args)
    if bboxes is None: return img

    # Object Recognition
    face_ids, probs, _ = ML.Recognition(img, bboxes, args, model_args)

    # Mosaic
    if args["DO_MOSAIC"]:
        processed_img = Mosaic(img, bboxes, face_ids, n=10)

    # 특정인에 bbox와 name을 보여주고 싶으면
    if args["DO_STROKE"]:
        processed_img = DrawRectImg(processed_img, bboxes, face_ids)

    return processed_img


def ProcessVideo(img, args, model_args, id_name):
    # global id_name
    # Object Detection
    bboxes, probs = ML.Detection(img, args, model_args)
    
    # ML.DeepsortRecognition
    
    outputs = ML.Deepsort(img, bboxes, probs, model_args['Tracking'])
    # last_out = outputs 

    # if boxes is None:
    #     return img, outputs
    
    if len(outputs) > 0:
        bbox_xyxy = outputs[:, :4]
        identities = outputs[:, -1]
        if identities[-1] not in id_name.keys():
            # Update가 생기거나
            id_name, probs, face_embedding = ML.Recognition(img, bbox_xyxy, args, model_args, id_name, identities)
            # faces_img, embedding_data = face_embedding.get_data()
            # print('face image 데이터와 embedding 데이터!', faces_img.shape, embedding_data.shape)
        elif all(['unknown' == i for i in id_name.values()]):
            # 모든 대상이 unknown tag일 경우
            id_name, probs, face_embedding = ML.Recognition(img, bbox_xyxy, args, model_args, id_name, identities, reset=True)

        if args["DO_MOSAIC"]:
            processed_img = Mosaic(img, bbox_xyxy, identities, 10, id_name)
    
        # 특정인에 bbox와 name을 보여주고 싶으면
        if args["DO_STROKE"]:
            processed_img = DrawRectImg(processed_img, bbox_xyxy, identities, id_name)
    else:
        processed_img = img
    
    return processed_img, id_name
