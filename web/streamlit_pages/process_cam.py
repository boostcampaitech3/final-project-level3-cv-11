import av
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import torch

from configs.server import rtc_configuration
from web.fastapi_engine.model_pipeline import init_model_args, ProcessImage, ProcessVideo
from web.fastapi_engine.database import load_face_db
from web.fastapi_engine.model_assignment import assign_detector, assign_recognizer, assign_tracker

current_frame = None
id_name = {}

class VideoProcessor:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
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
        
        self.model_args = dict()
        self.model_args["Device"] = self.device
        self.model_args["Detection"] = assign_detector(self.args["WHICH_DETECTOR"], self.device)
        self.model_args["Recognition"] = assign_recognizer(self.args["WHICH_RECOGNIZER"], self.device)
        self.model_args['Tracking'] = assign_tracker(self.args["WHICH_TRACKER"], self.device)
        face_db_path = ".database/"
        self.model_args['Face_db'] = load_face_db(".assets/sample_input/test_images2", face_db_path, self.device, self.args, self.model_args)

    def recv(self, frame):
        # The frame is an instance of av.VideoFrame (or av.AudioFrame when dealing with audio) of PyAV library.
        #   - https://pyav.org/docs/develop/api/video.html#av.video.frame.VideoFrame
        global current_frame, id_name
        current_frame = frame

        ndarray_img = frame.to_ndarray(format="bgr24")
        # processed_frame = ProcessImage(ndarray_img, self.args, self.model_args)
        processed_frame, id_name = ProcessVideo(ndarray_img, self.args, self.model_args, id_name)
        
        return av.VideoFrame.from_ndarray(processed_frame, format="bgr24")

    def save_current_frame():
        global current_frame
        frame_ndarray = current_frame.to_ndarray(format="bgr24")
        cv2.imwrite(".result_output/cam_current_frame.jpg", frame_ndarray)


def app():
    # 타겟 데이터가 웹캠인 경우 활성화 되는 페이지
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
        st.image(".result_output/cam_current_frame.jpg")
    