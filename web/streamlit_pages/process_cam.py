import av
import cv2
import numpy as np
import requests
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import torch

from configs.server import rtc_configuration
from web.fastapi_engine.model_pipeline import init_model_args
from web.fastapi_engine.detection import load_face_db
from web.fastapi_engine import ml_part as ML
from web.fastapi_engine.util import Mosaic, DrawRectImg
from web.fastapi_engine.model_assignment import assign_detector, assign_recognizer

current_frame = None

class VideoProcessor:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.args = dict()
        self.args["PROCESS_TARGET"] = "img"
        self.args['DEBUG_MODE'] = False
        self.args["BBOX_THRESHOLD"] = 30
        self.args["RECOG_THRESHOLD"] = 0.8
        self.args["DO_DETECTION"] = True
        self.args["WHICH_DETECTOR"] = "MTCNN"
        self.args["DO_RECOGNITION"] = True
        self.args["WHICH_RECOGNIZER"] = "FaceNet"
        self.args["DO_TRACKING"] = False
        
        self.model_args = dict()
        self.model_args["Device"] = self.device
        self.model_args["Detection"] = assign_detector("MTCNN", self.device)
        self.model_args["Recognition"] = assign_recognizer("FaceNet", self.device)
        face_db_path = ".database/face_db"
        self.model_args['Face_db'] = load_face_db(".assets/sample_input/test_images", face_db_path, ".database/img_db", self.device, self.args, self.model_args)

    def recv(self, frame):
        # The frame is an instance of av.VideoFrame (or av.AudioFrame when dealing with audio) of PyAV library.
        #   - https://pyav.org/docs/develop/api/video.html#av.video.frame.VideoFrame
        global current_frame
        current_frame = frame
            
        ndarray_img = frame.to_ndarray(format="rgb24")
        processed_frame = self.ProcessImage(ndarray_img, self.args, self.model_args)
        
        return av.VideoFrame.from_ndarray(processed_frame, format="rgb24")

    def save_current_frame():
        global current_frame
        frame_ndarray = current_frame.to_ndarray(format="bgr24")
        cv2.imwrite(".result_output/cam_current_frame.jpg", frame_ndarray)
    
    def ProcessImage(self, img, args, model_args):
        process_target = args["PROCESS_TARGET"]

        # Object Detection
        bboxes = ML.Detection(img, args, model_args)
        print(f"{len(bboxes)}개 얼굴 찾음!")
        if bboxes is None:
            # if args["WHICH_DETECTOR"] == "MTCNN":
            #     if process_target == "vid": # torchvision
            #         img = img.numpy()
            #     # Color channel: RGB -> BGR
            #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img

        # Object Recognition
        face_ids = ML.Recognition(img, bboxes, args, model_args)
        # face_ids = [ 'unknown' for _ in bboxes ]

        # if args["WHICH_DETECTOR"] == "MTCNN":
        #     # 모자이크 전처리
        #     if process_target == "vid": # torchvision
        #         img = img.numpy()
        #     # Color channel: RGB -> BGR
        #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Mosaic
        img = Mosaic(img, bboxes, face_ids, n=10)

        # 특정인에 bbox와 name을 보여주고 싶으면
        processed_img = DrawRectImg(img, bboxes, face_ids)

        return processed_img


def app():
    # 타겟 데이터가 웹캠인 경우 활성화 되는 페이지
    # st.session_state.process_target = "cam"
    # st.session_state.save_face_embedding = False
    
    # st.text(""); st.text("") # 공백
    # st.markdown("###### 알고리즘 선택")
    # choice_col1, choice_col2, choice_col3 = st.columns(3)
    # st.session_state.which_detector = choice_col1.selectbox(
    #     "Detection model", 
    #     ("MTCNN", "RetinaFace", "YOLOv5", "Facesbox", "HaarCascades"), 
    #     index=2
    # )
    # st.session_state.which_recognizer = choice_col2.selectbox(
    #     "Recognition model", 
    #     ("FaceNet", "ArcFace", "ArcFace_Mofy"), 
    #     index=0
    # )
    # st.session_state.which_tracker = choice_col3.selectbox(
    #     "Tracking algorithm", 
    #     ("DeepSort", ), 
    #     index=0
    # )
    # st.session_state.do_detection = True
    # st.session_state.do_recognition = True
    # st.session_state.do_tracking = True
    
    st.text(""); st.text("") # 공백
    st.markdown("###### 실시간 웹캠 영상 처리")
    
#     args = {
#         "PROCESS_TARGET": st.session_state.process_target,
#         "SAVE_FACE_EMBEDDING": st.session_state.save_face_embedding,
        
#         "DO_DETECTION": st.session_state.do_detection,
#         # "WHICH_DETECTOR": st.session_state.which_detector,
#         "DO_RECOGNITION": st.session_state.do_recognition,
#         # "WHICH_RECOGNIZER": st.session_state.which_recognizer,
#         "DO_TRACKING": st.session_state.do_tracking,
#         # "WHICH_TRACKER": st.session_state.which_tracker
#     }
#     r = requests.post("http://localhost:8001/settings", json=args)
    # st.write(r)
    
    webrtc_streamer(
        key="webcam", video_processor_factory=VideoProcessor,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

    if st.button("click", on_click=VideoProcessor.save_current_frame):
        st.image(".result_output/cam_current_frame.jpg")
    