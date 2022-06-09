import cv2
import datetime
import hashlib
import os
import requests
from shutil import rmtree
import streamlit as st
from streamlit_webrtc import webrtc_streamer

from configs.server import rtc_configuration


current_frame = None

class VideoProcessor:
    def __init__(self):
        pass

    def recv(self, frame):
        # The frame is an instance of av.VideoFrame (or av.AudioFrame when dealing with audio) of PyAV library.
        #   - https://pyav.org/docs/develop/api/video.html#av.video.frame.VideoFrame
        global current_frame
        current_frame = frame
        
        return frame

    def save_current_frame():
        global current_frame
        frame_ndarray = current_frame.to_ndarray(format="bgr24")
        
        if not os.path.exists(st.session_state.output_dir):
            os.mkdir(st.session_state.output_dir)
        cv2.imwrite(st.session_state.output_dir + "input.jpg", frame_ndarray)


def app():
    # DB에 얼굴 사진을 등록
    if "target_type" not in st.session_state:
        st.session_state.target_type = None # 초기값
    
    st.text(""); st.text("") # 공백
    st.markdown("###### 데이터베이스에 얼굴을 등록")

    st.text(""); st.text("") # 공백
    uploaded_name = st.text_input("등록할 사람의 이름을 입력하세요", value="guest")
    
    st.text("") # 공백
    st.text("등록할 사진의 업로드 방식을 선택하세요")
    col1, col2 = st.columns(2)
    button1 = col1.button("사진 업로드")
    button2 = col2.button("실시간 웹캠")
    
    if button1:
        st.session_state.target_type = "IMAGE"
    if button2:
        st.session_state.target_type = "WEBCAM"
    
    if st.session_state.target_type is None:
        pass
    else:
        now = datetime.datetime.now()
        shastr = hashlib.sha224(str(now).encode()).hexdigest()[:8]
        st.session_state.output_dir = f".result_output/{shastr}/"
        
        if st.session_state.target_type == "IMAGE":
            uploaded_file = st.file_uploader("등록할 사람의 사진을 업로드하세요", type=["png", "jpg", "jpeg"])
            if uploaded_file:
                uploaded_file_type = uploaded_file.name.split('.')[-1]
                if not os.path.exists(st.session_state.output_dir):
                    os.mkdir(st.session_state.output_dir)
                with open(st.session_state.output_dir + f"input.{uploaded_file_type}", "wb") as picfile:
                    picfile.write(uploaded_file.getvalue())
                input_file_name = st.session_state.output_dir + f"input.{uploaded_file_type}"
                output_file_name = st.session_state.output_dir + f"output.{uploaded_file_type}"
            
        else: # if st.session_state.target_type == "WEBCAM":
            webrtc_streamer(
                key="webcam", video_processor_factory=VideoProcessor,
                rtc_configuration=rtc_configuration,
                media_stream_constraints={"video": True, "audio": False}
            )
            if st.button("캡쳐!", on_click=VideoProcessor.save_current_frame):
                input_file_name = st.session_state.output_dir + "input.jpg"
                output_file_name = st.session_state.output_dir + "output.jpg"
        
        if os.path.exists(st.session_state.output_dir):
            st.text(""); st.text("") # 공백
            st.markdown("###### 데이터 처리 결과")
            
            args = {
                "USERNAME": st.session_state.username,
                "REQUEST_ID": shastr,

                "INPUT_FILE_NAME": input_file_name,
                "OUTPUT_FILE_NAME": output_file_name,
                
                "SAVE_FACE_NAME": uploaded_name
            }
            response = requests.post("http://localhost:8001/update_db", json=args)
            result_str = response.json()["products"][0]["result"]
            if result_str == "Success":
                st.write(f"{uploaded_name} 님의 얼굴을 저장 성공했습니다.")
                rmtree(st.session_state.output_dir)
            else:
                st.write("저장 실패했습니다. 다시 시도해주세요.")
