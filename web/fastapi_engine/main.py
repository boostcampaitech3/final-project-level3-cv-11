import cv2
from datetime import datetime
import io
import numpy as np
from PIL import Image
from typing import List, Union, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File
from fastapi.param_functions import Depends
from pydantic import BaseModel, Field
from uuid import UUID, uuid4

import torch
import torchvision

from configs.process import params as init_params
from web.fastapi_engine.model_assignment import assign_detector, assign_recognizer, assign_tracker
from web.fastapi_engine.model_pipeline import init_model_args, ProcessImage


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
    WHICH_DETECTOR: str = "MTCNN" # "YOLOv5"
    COLOR_DETECTOR: str = "RGB"
    
    DO_RECOGNITION: bool = False
    WHICH_RECOGNIZER: str = "FaceNet"
    
    SAVE_FACE_EMBEDDING: bool = False
    SAVE_FACE_NAME: str = "guest"
    
    DO_TRACKING: bool = False
    WHICH_TRACKER: str = "DeepSort"
    
    DEBUG_MODE: bool = False
    

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


@app.post("/order", description="입력된 자료를 처리합니다.")
async def process_target(
        files: List[UploadFile] = File(...)
    ):
    global model_detection, model_recognition, algo_tracking
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
        
        with open(".result_output/input_video.mp4", "wb") as fp:
            fp.write(video_bytes)
        video = torchvision.io.VideoReader(".result_output/input_video.mp4", stream='video')
        if app.state.args['DEBUG_MODE']:
            print(video.get_metadata())
        video.set_current_stream('video')
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(".result_output/output_video.mp4", fourcc, 24.0, (1280,720))
        for frame in video:
            img = frame['data']
            # img.to(model_args['Device'])
            img = torch.permute(img, (1, 2, 0))
            img = ProcessImage(img, app.state.args, model_args)

            out.write(img)
        out.release()
        
        product = DetectionResult(result="Success")
        products.append(product)
    # =================== Video =======================
    
    elif app.state.args["PROCESS_TARGET"] == "cam":
        pass
    
    all_result = ResultList(products=products)
    return all_result
