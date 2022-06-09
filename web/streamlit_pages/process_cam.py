import av
import cv2
import torch
import os
import shutil
import numpy as np

from torchvision.transforms import functional as F

import streamlit as st
from streamlit_webrtc import webrtc_streamer

from configs.server import rtc_configuration

from web.fastapi_engine.detection import get_embeddings
from web.fastapi_engine.util import crop_resize
from web.fastapi_engine.model_pipeline import init_model_args, ProcessVideo
from web.fastapi_engine import ml_part as ML
from web.fastapi_engine.database import load_face_db
from web.fastapi_engine.model_assignment import assign_detector, assign_recognizer, assign_tracker

current_frame = None
id_name = {}
args = {}
model_args = {}


class VideoProcessor:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        global args, model_args

        self.args = dict()
        self.args["PROCESS_TARGET"] = "img"
        self.args['DEBUG_MODE'] = False
        self.args["BBOX_THRESHOLD"] = 30
        self.args["RECOG_THRESHOLD"] = 0.8
        self.args["DO_DETECTION"] = True
        self.args["WHICH_DETECTOR"] = "YOLOv5"
        self.args["DO_RECOGNITION"] = True
        self.args["WHICH_RECOGNIZER"] = "FaceNet"
        self.args["DO_TRACKING"] = True
        self.args["WHICH_TRACKER"] = "DeepSort"
        args = self.args

        self.model_args = dict()
        self.model_args["Device"] = self.device
        self.model_args["Detection"] = assign_detector(self.args["WHICH_DETECTOR"], self.device)
        self.model_args["Recognition"] = assign_recognizer(self.args["WHICH_RECOGNIZER"], self.device)
        self.model_args['Tracking'] = assign_tracker(self.args["WHICH_TRACKER"], self.device)
        face_db_path = ".database/"
        self.model_args['Face_db'] = load_face_db(".assets/sample_input/test_images2", face_db_path, self.device, self.args, self.model_args)
        model_args = self.model_args


    def recv(self, frame):
        # The frame is an instance of av.VideoFrame (or av.AudioFrame when dealing with audio) of PyAV library.
        #   - https://pyav.org/docs/develop/api/video.html#av.video.frame.VideoFrame
        global current_frame, id_name
        current_frame = frame

        ndarray_img = frame.to_ndarray(format="bgr24")
        # processed_frame = ProcessImage(ndarray_img, self.args, self.model_args)
        id_name = {}
        processed_frame, id_name = ProcessVideo(ndarray_img, self.args, self.model_args, id_name)
        print(list(self.model_args['Face_db'].keys()))
        return av.VideoFrame.from_ndarray(processed_frame, format="bgr24")

    def save_current_frame():
        global current_frame, args, model_args
        frame_ndarray = current_frame.to_ndarray(format="bgr24")
        output_path = '.result_output/cam/'

        if os.path.isdir(output_path):
            shutil.rmtree(output_path)
        
        os.mkdir(output_path)

        bboxes, probs = ML.Detection(frame_ndarray, args, model_args)
        if len(bboxes) > 5:
            bboxes = bboxes[:5]
            probs = probs[:5]
        
        for i, bbox in enumerate(bboxes):
            bbox = np.round(bbox).astype(int)
            crop_img = crop_resize(frame_ndarray, bbox, 256)
            cv2.imwrite(os.path.join(output_path, f'capture_{str(i)}.jpg'), crop_img)
        

def app():
    # 타겟 데이터가 웹캠인 경우 활성화 되는 페이지

    def db_button(i):
        global model_args

        print(f'button {i} click!')

        for j in range(1000):
            if j not in model_args['Face_db'].keys():
                nm = str(j)
                break
            
        face = cv2.imread('.result_output/cam/' + f'capture_{str(i)}.jpg')
        face = F.to_tensor(np.float32(face))
        face = (face - 127.5) / 128.0
        face = torch.stack([face])
        face = face.to(model_args["Device"])
        embeddings = model_args['Recognition'](face).detach().cpu().numpy()
        model_args['Face_db'][nm] = [embeddings]
        
        print('db add complete!')


    st.session_state.save_face_embedding = False
    
    st.text(""); st.text("") # 공백
    st.markdown("###### 실시간 웹캠 영상 처리")
    
    webrtc_streamer(
        key="webcam", video_processor_factory=VideoProcessor,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

    if st.button("click", on_click=VideoProcessor.save_current_frame):
        img_list = os.listdir('.result_output/cam/')
        n = len(img_list)
        st.text(""); st.text("") # 공백
        st.markdown("###### 데이터 베이스에 추가할 사람을 선택해주세요.")
        
        col = st.columns(n)
        for i in range(n):
            si = str(i)
            col[i].header(f'image {si}')
            col[i].button("선택", on_click=db_button, args=[i], key=i)
            col[i].image(".result_output/cam/" + f'capture_{str(i)}.jpg')

        