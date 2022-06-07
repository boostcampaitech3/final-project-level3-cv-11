import cv2
import torch

from web.fastapi_engine.detection import load_face_db
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
            face_db_path = ".database/face_db"
            if args["WHICH_DETECTOR"] == "RetinaFace":
                face_db_path += "_BGR"

            face_db = load_face_db(".assets/sample_input/test_images", face_db_path, ".database/img_db", device, args, model_args)

            model_args['Face_db'] = face_db
            
            if args["DO_TRACKING"]:
                # 3. Load Tracking Algorithm
                model_args["Tracking"] = algo_tracking
    
    return model_args


def ProcessImage(img, args, model_args):
    process_target = args["PROCESS_TARGET"]

    # Object Detection
    bboxes = ML.Detection(img, args, model_args)
    # print(f"{len(bboxes)}개 얼굴 찾음!")
    if bboxes is None:
        if args["WHICH_DETECTOR"] == "MTCNN":
            if process_target == "vid": # torchvision
                img = img.numpy()
            # Color channel: RGB -> BGR
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    # Object Recognition
    face_ids = ML.Recognition(img, bboxes, args, model_args)

    if args["WHICH_DETECTOR"] == "MTCNN":
        # 모자이크 전처리
        if process_target == "vid": # torchvision
            img = img.numpy()
        # Color channel: RGB -> BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Mosaic
    img = Mosaic(img, bboxes, face_ids, n=10)

    # 특정인에 bbox와 name을 보여주고 싶으면
    processed_img = DrawRectImg(img, bboxes, face_ids)

    return processed_img
