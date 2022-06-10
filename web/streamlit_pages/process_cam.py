import av
import cv2
import datetime
from glob import glob
import hashlib
import numpy as np
import os
import pickle
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import torch

from configs.server import rtc_configuration
from web.fastapi_engine.model_pipeline import init_model_args, ProcessImage, ProcessVideo
from web.fastapi_engine.model_assignment import assign_detector, assign_recognizer, assign_tracker
from web.fastapi_engine.database import load_face_db
from web.fastapi_engine import ml_part as ML

current_frame = None
id_name = {}

update_info = {
    "prev_name": list(),
    "new_name": list()
}

init_info = {
    "username": None,
    "do_detection": None,
    "which_detector": None,
    "do_recognition": None,
    "which_recognizer": None,
    "do_tracking": None,
    "which_tracker": None,
    "do_mosaic": None,
    "do_stroke": None
}

self_info = {
    "args": None,
    "model_args": None
}

class VideoProcessor:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.args = dict()
        self.args["PROCESS_TARGET"] = "vid"
        self.args['DEBUG_MODE'] = False
        self.args["BBOX_THRESHOLD"] = 30
        self.args["RECOG_THRESHOLD"] = 0.8
        self.args["DO_DETECTION"] = init_info["do_detection"]
        self.args["WHICH_DETECTOR"] = init_info["which_detector"]
        self.args["DO_RECOGNITION"] = init_info["do_recognition"]
        self.args["WHICH_RECOGNIZER"] = init_info["which_recognizer"]
        self.args["DO_TRACKING"] = init_info["do_tracking"]
        self.args["WHICH_TRACKER"] = init_info["which_tracker"]
        self.args["DO_MOSAIC"] = init_info["do_mosaic"]
        self.args["DO_STROKE"] = init_info["do_stroke"]
        
        self.model_args = dict()
        self.model_args["Device"] = self.device
        self.model_args["Detection"] = assign_detector(self.args["WHICH_DETECTOR"], self.device)
        self.model_args["Recognition"] = assign_recognizer(self.args["WHICH_RECOGNIZER"], self.device)
        self.model_args['Tracking'] = assign_tracker(self.args["WHICH_TRACKER"], self.device)
        self.model_args['Face_db_path'] = f".database/{init_info['username']}/{self.args['WHICH_DETECTOR']}"  # st.session_state.username
        self.model_args['Face_db'] = load_face_db(self.model_args['Face_db_path'])
        
        self_info["args"] = self.args
        self_info["model_args"] = self.model_args

    def recv(self, frame):
        # The frame is an instance of av.VideoFrame (or av.AudioFrame when dealing with audio) of PyAV library.
        #   - https://pyav.org/docs/develop/api/video.html#av.video.frame.VideoFrame
        global current_frame, id_name, update_info
        current_frame = frame

        ndarray_img = frame.to_ndarray(format="bgr24")
        # processed_frame = ProcessImage(ndarray_img, self.args, self.model_args)
        processed_frame, id_name = ProcessVideo(ndarray_img, self.args, self.model_args, id_name)
        
        if len(update_info["prev_name"]) != 0:
            self.update_face_db()
        
        return av.VideoFrame.from_ndarray(processed_frame, format="bgr24")

    def detect_current_frame():
        global current_frame
        frame_ndarray = current_frame.to_ndarray(format="bgr24")
        
        now = datetime.datetime.now()
        shastr = hashlib.sha224(str(now).encode()).hexdigest()[:8]
        st.session_state.output_dir = f".result_output/{shastr}/"
        
        if not os.path.exists(st.session_state.output_dir):
            os.mkdir(st.session_state.output_dir)
        
        self_args = self_info["args"]
        self_args["PROCESS_TARGET"] = "img"
        self_model_args = self_info["model_args"]
        
        # Object Detection
        bboxes, probs = ML.Detection(frame_ndarray, self_args, self_model_args)
        if bboxes is None: return 

        # Object Recognition
        _, _, face_embeddings = ML.Recognition(frame_ndarray, bboxes, self_args, self_model_args)
        faces_img, embedding_data = face_embeddings.get_data()
        
        for face, emb in zip(faces_img, embedding_data):
            face = torch.permute(face, (1,2,0)).contiguous().detach().cpu().numpy()
            face *= 255
            shastr = hashlib.sha224(face).hexdigest()[:8]
        
            emb = emb.numpy()
            if shastr in self_model_args['Face_db']:
                self_model_args['Face_db'][shastr].append(emb)
            else:
                self_model_args['Face_db'][shastr] = [emb]

            with open(os.path.join(self_model_args['Face_db_path'], "tmp_db"), "wb") as f:
                pickle.dump(self_model_args['Face_db'], f)
        
            cv2.imwrite(st.session_state.output_dir + f"{shastr}.jpg", face)
    
    def update_face_db(self):
        global update_info
        with open(os.path.join(self.model_args['Face_db_path'], "tmp_db"), "rb") as f:
            tmp_db = pickle.load(f)
        with open(os.path.join(self.model_args['Face_db_path'], "face_db"), "rb") as f:
            face_db = pickle.load(f)
        
        for p, n in zip(update_info["prev_name"], update_info["new_name"]):
            face_db[n] = tmp_db[p]
        
        # face_db 업데이트
        with open(os.path.join(self.model_args['Face_db_path'], "face_db"), "wb") as f:
            pickle.dump(face_db, f)
        self.model_args['Face_db'] = face_db
        
        # tmp 삭제 및 update_info 초기화
        os.remove(os.path.join(self.model_args['Face_db_path'], "tmp_db"))
        update_info = {
            "prev_name": list(),
            "new_name": list()
        }


def app(parent_state):
    # 상속
    if init_info["username"] != parent_state.username:
        init_info["username"] = parent_state.username
    if init_info["do_detection"] != parent_state.do_detection:
        init_info["do_detection"] = parent_state.do_detection
    if init_info["which_detector"] != parent_state.which_detector:
        init_info["which_detector"] = parent_state.which_detector
    if init_info["do_recognition"] != parent_state.do_recognition:
        init_info["do_recognition"] = parent_state.do_recognition
    if init_info["which_recognizer"] != parent_state.which_recognizer:
        init_info["which_recognizer"] = parent_state.which_recognizer
    if init_info["do_tracking"] != parent_state.do_tracking:
        init_info["do_tracking"] = parent_state.do_tracking
    if init_info["which_tracker"] != parent_state.which_tracker:
        init_info["which_tracker"] = parent_state.which_tracker
    if init_info["do_mosaic"] != parent_state.do_mosaic:
        init_info["do_mosaic"] = parent_state.do_mosaic
    if init_info["do_stroke"] != parent_state.do_stroke:
        init_info["do_stroke"] = parent_state.do_stroke
    
    # 타겟 데이터가 웹캠인 경우 활성화 되는 페이지
    
    st.text(""); st.text("") # 공백
    st.markdown("###### 실시간 웹캠 영상 처리")
    
    webrtc_streamer(
        key="webcam", video_processor_factory=VideoProcessor,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

    st.text("") # 공백
    if st.button("캡쳐!", on_click=VideoProcessor.detect_current_frame):
        image_list = glob(st.session_state.output_dir + "*.jpg")
        
        all_txts = list()
        all_shas = list()
        
        idx = 0
        while len(image_list) > 0:
            cols = st.columns(5)
            for i in range(5):
                image_path = image_list.pop(0)
                chk = cols[i].checkbox("등록 여부")
                txt = cols[i].text_input("등록할 이름", value=image_path[-12:-4])
                cols[i].image(image_path, use_column_width=True)
                
                all_txts.append(txt)
                all_shas.append(image_path[-12:-4])
                i += 1
                
                if len(image_list) == 0:
                    break
        
        st.text("") # 공백
        if st.button("DB 업데이트"):
            global update_info
            s_list = list()
            t_list = list()
            for t, s in zip(all_txts, all_shas):
                if t != s: # 변경된 항목들에 대해서만 업데이트
                    s_list.append(s)
                    t_list.append(t)
            
            # 한꺼번에 업데이트
            update_info["new_name"] = t_list
            update_info["prev_name"] = s_list
    