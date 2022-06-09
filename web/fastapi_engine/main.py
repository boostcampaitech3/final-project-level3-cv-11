import cv2
import torch

from time import time
from datetime import datetime
import numpy as np
from typing import List, Union, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel, Field
from uuid import UUID, uuid4

import torch
import torchvision

from configs.process import params as init_params
from web.fastapi_engine.model_assignment import assign_detector, assign_recognizer, assign_tracker
from web.fastapi_engine.model_pipeline import init_model_args, SaveSingleEmbedding, ProcessImage, ProcessVideo

from moviepy.editor import VideoFileClip

app = FastAPI()
app.state.args = dict()

model_detection = None
model_recognition = None
algo_tracking = None


@app.get("/")
def hello_world():
    return {"hello": "world"}

class FastAPIArgs(BaseModel):
    PROCESS_TARGET: str = "img"
    
    DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    DO_DETECTION: bool = False
    WHICH_DETECTOR: str = "YOLOv5"
    COLOR_DETECTOR: str = "RGB"
    
    DO_RECOGNITION: bool = False
    WHICH_RECOGNIZER: str = "FaceNet"
    RECOG_THRESHOLD: float = 0.8
    
    DO_TRACKING: bool = False
    WHICH_TRACKER: str = "DeepSort"
    
    USERNAME: str = "guest"
    
    DEBUG_MODE: bool = False

class RequestArgs(BaseModel):
    USERNAME: str
    REQUEST_ID: str
    
    DO_MOSAIC: bool = False
    DO_STROKE: bool = False
    
    INPUT_FILE_NAME: str = "input.png"
    OUTPUT_FILE_NAME: str = "output.png"
    
    SAVE_FACE_NAME: str = "guest"

class DetectionResult(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    result: str

class ResultList(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    products: List[DetectionResult] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    

#########################

@app.post("/settings", description="환경설정을 진행합니다.")
async def set_args(
        json: FastAPIArgs
    ):
    global model_detection, model_recognition, algo_tracking
    
    args = json.dict()
    for k in args:
        app.state.args[k] = args[k]
    for k in init_params:
        if k not in app.state.args:
            app.state.args[k] = init_params[k]
    
    if app.state.args["DO_DETECTION"] == False:
        model_detection = None
    else:
        model_detection = assign_detector(app.state.args["WHICH_DETECTOR"], app.state.args["DEVICE"])
        if app.state.args["WHICH_DETECTOR"] == "MTCNN":
            app.state.args["COLOR_DETECTOR"] = "RGB"
        elif app.state.args["WHICH_DETECTOR"] == "RetinaFace":
            app.state.args["COLOR_DETECTOR"] = "BGR"
        elif app.state.args["WHICH_DETECTOR"] == "YOLOv5":
            app.state.args["COLOR_DETECTOR"] = "BGR"
        elif app.state.args["WHICH_DETECTOR"] == "FaceBoxes":
            app.state.args["COLOR_DETECTOR"] = "RGB"
        elif app.state.args["WHICH_DETECTOR"] == "HaarCascades":
            app.state.args["COLOR_DETECTOR"] = "GRAY"
        else:
            raise NotImplementedError(app.state.args["WHICH_DETECTOR"])
    
    if app.state.args["DO_RECOGNITION"] == False:
        model_recognition = None
    else:
        model_recognition = assign_recognizer(app.state.args["WHICH_RECOGNIZER"], app.state.args["DEVICE"])
    
    if app.state.args["DO_TRACKING"] == False:
        algo_tracking = None
    else:
        algo_tracking = assign_tracker(app.state.args["WHICH_TRACKER"], app.state.args["DEVICE"])

        
@app.post("/update_db", description="타겟을 임베딩으로 변환하여 저장합니다.")
async def save_target(
        json: RequestArgs
        # files: List[UploadFile] = File(...)
    ):
    global model_detection, model_recognition
    
    args = json.dict()
    for k in args:
        app.state.args[k] = args[k]
    
    model_args = init_model_args(app.state.args, model_detection, model_recognition, None)
    
    products = []
    
    if app.state.args["PROCESS_TARGET"] == "img":
        # Color channel: BGR
        img = cv2.imread(app.state.args["INPUT_FILE_NAME"])
        
        start = time()
        SaveSingleEmbedding(img, app.state.args, model_args)
        
        print(f'time: {time() - start}')
        print('done.')
        
        product = DetectionResult(result="Success")
        products.append(product)
    
    else:
        raise ValueError(app.state.args["PROCESS_TARGET"])
    
    all_result = ResultList(products=products)
    return all_result

@app.post("/order", description="타겟을 모자이크 처리합니다.")
async def process_target(
        json: RequestArgs
        # files: List[UploadFile] = File(...)
    ):
    global model_detection, model_recognition, algo_tracking
    
    args = json.dict()
    for k in args:
        app.state.args[k] = args[k]
    
    model_args = init_model_args(app.state.args, model_detection, model_recognition, algo_tracking)
    
    products = []
    
    # =================== Image =======================
    if app.state.args["PROCESS_TARGET"] == "img":
        # Color channel: BGR
        img = cv2.imread(app.state.args["INPUT_FILE_NAME"])
        
        start = time()
        
        # if app.state.args["COLOR_DETECTOR"] == "RGB":
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # elif app.state.args["COLOR_DETECTOR"] == "BGR":
        #     pass
        # elif app.state.args["COLOR_DETECTOR"] == "GRAY":
        #     img = cv2.imdecode(img, cv2.COLOR_BGR2GRAY)
        # else:
        #     raise ValueError(app.state.args["COLOR_DETECTOR"])
        
        
        img = ProcessImage(img, app.state.args, model_args)
        
        cv2.imwrite(app.state.args["OUTPUT_FILE_NAME"], img)
        
        print(f'time: {time() - start}')
        print('done.')
        
        product = DetectionResult(result="Success")
        products.append(product)
    # =================== Image =======================
    
    # =================== Video =======================
    elif app.state.args["PROCESS_TARGET"] == "vid":
        clip = VideoFileClip(app.state.args["INPUT_FILE_NAME"])
        audio_data = clip.audio

        cap = cv2.VideoCapture(app.state.args["INPUT_FILE_NAME"])
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        width = int(cap.get(3))
        height = int(cap.get(4))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(app.state.args["OUTPUT_FILE_NAME"], fourcc, fps, (width, height))
        # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        id_name = {}
        start = time()
        while True:
            ret, img = cap.read()
            # Color channel: BGR
            if ret:
                if app.state.args["DO_TRACKING"]:
                    # if app.state.args["COLOR_DETECTOR"] == "RGB":
                    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # elif app.state.args["COLOR_DETECTOR"] == "BGR":
                    #     pass
                    # elif app.state.args["COLOR_DETECTOR"] == "GRAY":
                    #     img = cv2.imdecode(img, cv2.COLOR_BGR2GRAY)
                    # else:
                    #     raise ValueError(app.state.args["COLOR_DETECTOR"])
                    
                    img, id_name = ProcessVideo(img, app.state.args, model_args, id_name)
                else:
                    img = ProcessImage(img, app.state.args, model_args)
                out.write(img)
            else:
                break

        cap.release()
        out.release()
        
        video = VideoFileClip(app.state.args["OUTPUT_FILE_NAME"])
        output = video.set_audio(audio_data)
        output.write_videofile(app.state.args["OUTPUT_FILE_NAME"])

        print(f'time: {time() - start}')
        print(f'original video fps: {fps}')
        print('done.')

        product = DetectionResult(result="Success")
        products.append(product)
    # =================== Video =======================
    
    else:
        raise NotImplementedError(app.state.args["PROCESS_TARGET"])
    
    all_result = ResultList(products=products)
    return all_result
