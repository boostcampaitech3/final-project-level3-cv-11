import av
import cv2
import datetime
from glob import glob
import hashlib
import os
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

class VideoProcessor:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.args = dict()
        self.args["PROCESS_TARGET"] = "img"
        self.args['DEBUG_MODE'] = False
        self.args["BBOX_THRESHOLD"] = 30
        self.args["RECOG_THRESHOLD"] = 0.8
        self.args["DO_DETECTION"] = True # st.session_state.do_detection
        self.args["WHICH_DETECTOR"] = "YOLOv5" # st.session_state.which_detector
        self.args["DO_RECOGNITION"] = True # st.session_state.do_recognition
        self.args["WHICH_RECOGNIZER"] = "FaceNet" # st.session_state.which_recognizer
        self.args["DO_TRACKING"] = True # st.session_state.do_tracking
        self.args["WHICH_TRACKER"] = "DeepSort" # st.session_state.which_tracker
        
        self.model_args = dict()
        self.model_args["Device"] = self.device
        self.model_args["Detection"] = assign_detector(self.args["WHICH_DETECTOR"], self.device)
        self.model_args["Recognition"] = assign_recognizer(self.args["WHICH_RECOGNIZER"], self.device)
        self.model_args['Tracking'] = assign_tracker(self.args["WHICH_TRACKER"], self.device)
        self.model_args['Face_db_path'] = f".database/{'sunghyu'}/{self.args['WHICH_DETECTOR']}" # st.session_state.username
        self.model_args['Face_db'] = load_face_db(self.model_args['Face_db_path'])

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
            
        # Object Detection
        bboxes, probs = ML.Detection(frame_ndarray, self.args, self.model_args)
        if bboxes is None: return 

        # Object Recognition
        _, _, face_embeddings = ML.Recognition(frame_ndarray, bboxes, self.args, self.model_args)
        faces_img, embedding_data = face_embeddings.get_data()
        
        for face, emb in zip(faces_img, embedding_data):
            shastr = hashlib.sha224(face).hexdigest()[:8]
        
            emb = emb.numpy()
            if shastr in self.model_args['Face_db']:
                self.model_args['Face_db'][shastr].append(emb)
            else:
                self.model_args['Face_db'][shastr] = [emb]

            with open(os.path.join(self.model_args['Face_db_path'], "tmp_db"), "wb") as f:
                pickle.dump(self.model_args['Face_db'], f)
        
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

    st.text("") # 공백
    if st.button("캡쳐!", on_click=VideoProcessor.detect_current_frame):
        image_list = glob(st.session_state.output_dir + "*.jpg")
        
        all_chks = list()
        all_txts = list()
        all_shas = list()
        
        idx = 0
        while len(image_list) > 0:
            cols = st.columns(5)
            for i in range(5):
                image_path = image_list.pop(0)
                chk = cols[i].checkbox("등록 여부")
                txt = cols[i].text_input("등록할 이름", value=image_path[-12:-4])
                cols[i].image(image_path)
                
                all_chks.append(chk)
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
            for c, t, s in zip(all_chks, all_txts, all_shas):
                if c:
                    s_list.append(s)
                    t_list.append(t)
            
            # 한꺼번에 업데이트
            update_info["new_name"] = t_list
            update_info["prev_name"] = s_list
    