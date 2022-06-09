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
from web.fastapi_engine.model_pipeline import init_model_args, ProcessImage, ProcessVideo

from moviepy.editor import VideoFileClip, AudioFileClip

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
            app.state.args["COLOR_DETECTOR"] = "RGB"
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
    file = files[0]
    
    # =================== Image =======================
    if app.state.args["PROCESS_TARGET"] == "img":
        image_bytes = await file.read()
        
        encoded_img = np.fromstring(image_bytes, dtype=np.uint8)
        if app.state.args["COLOR_DETECTOR"] == "RGB":
            image_array = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        elif app.state.args["COLOR_DETECTOR"] == "BGR":
            image_array = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
        elif app.state.args["COLOR_DETECTOR"] == "GRAY":
            image_array = cv2.imdecode(encoded_img, cv2.IMREAD_GRAYSCALE)
        else:
            raise ValueError(app.state.args["COLOR_DETECTOR"])
        
        processed_img = ProcessImage(image_array, app.state.args, model_args)
        
        cv2.imwrite(".result_output/output.png", processed_img)
        
        product = DetectionResult(result="Success")
        products.append(product)
    # =================== Image =======================
    
    # =================== Video =======================
    elif app.state.args["PROCESS_TARGET"] == "vid":
        video_bytes = await file.read()
        video_path = ".result_output/input_video.mp4"
        with open(video_path, "wb") as fp:
            fp.write(video_bytes)
        # sound
        clip = VideoFileClip(video_path)
        audio_data = clip.audio

        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(cap.get(3))
        height = int(cap.get(4))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter('.result_output/out_video.mp4', fourcc, fps, (width, height))
        # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        id_name = {}
        start = time()
        while True:
            ret, img = cap.read()
            # Color channel: BGR
            if ret:
                if app.state.args['DO_TRACKING']:
                    img, id_name = ProcessVideo(img, app.state.args, model_args, id_name)
                else:
                    img = ProcessImage(img, app.state.args, model_args)
                out.write(img)
            else:
                break

        cap.release()
        out.release()

        # sound
        video = VideoFileClip('.result_output/out_video.mp4')
        output = video.set_audio(audio_data)
        output.write_videofile('.result_output/output_video.mp4')

        print(f'original video fps: {fps}')
        print(f'time: {time() - start}')
        print('done.')

        product = DetectionResult(result="Success")
        products.append(product)
    # =================== Video =======================
    
    elif app.state.args["PROCESS_TARGET"] == "cam":
        pass
    
    all_result = ResultList(products=products)
    return all_result
