import cv2
import requests
import streamlit as st
from streamlit_webrtc import webrtc_streamer

from configs.server import rtc_configuration


current_frame = None
current_frame_bytes = None

class VideoProcessor:
    def __init__(self):
        pass

    def recv(self, frame):
        # The frame is an instance of av.VideoFrame (or av.AudioFrame when dealing with audio) of PyAV library.
        #   - https://pyav.org/docs/develop/api/video.html#av.video.frame.VideoFrame
        global current_frame
        current_frame = frame
        
        return frame

    def current_frame_to_bytes():
        global current_frame, current_frame_bytes
        frame_ndarray = current_frame.to_ndarray(format="bgr24")
        is_success, im_buf_arr = cv2.imencode(".jpg", frame_ndarray)
        current_frame_bytes = im_buf_arr.tobytes()

class DummyFile:
    name = "temp.jpg"
    type = "image/jpeg"


def app():
    # DB에 얼굴 사진을 등록
    st.session_state.process_target = "img"
    st.session_state.save_face_embedding = True
    
    if "target_type" not in st.session_state:
        st.session_state.target_type = None # 초기값
    
    st.text(""); st.text("") # 공백
    st.markdown("###### 데이터베이스에 얼굴을 등록")
    
    # st.text(""); st.text("") # 공백
    # st.markdown("###### 알고리즘 선택")
    # choice_col1, choice_col2, choice_col3 = st.columns(3)
    # st.session_state.which_detector = choice_col1.selectbox(
    #     "Detection model", 
    #     ("MTCNN", "RetinaFace", "YOLOv5", "FaceBoxes", "HaarCascades"), 
    #     index=2
    # )
    # st.session_state.which_recognizer = choice_col2.selectbox(
    #     "Recognition model", 
    #     ("FaceNet", "ArcFace", "ArcFace_Mofy"), 
    #     index=0
    # )

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
        
        if st.session_state.target_type == "IMAGE":
            uploaded_file = st.file_uploader("등록할 사람의 사진을 업로드하세요", type=["png", "jpg", "jpeg"])
            bytes_data = None if uploaded_file is None else uploaded_file.getvalue()
        else: # if st.session_state.target_type == "WEBCAM":
            webrtc_streamer(
                key="webcam", video_processor_factory=VideoProcessor,
                rtc_configuration=rtc_configuration,
                media_stream_constraints={"video": True, "audio": False}
            )
            if st.button("캡쳐!", on_click=VideoProcessor.current_frame_to_bytes):
                global current_frame_bytes
                bytes_data = current_frame_bytes
                uploaded_file = DummyFile()
            else:
                bytes_data = None
        
        if bytes_data:
            st.session_state.do_detection = True
            st.session_state.do_recognition = True

            st.text(""); st.text("") # 공백
            st.markdown("###### 데이터 처리 결과")

            args = {
                "process_target": st.session_state.process_target,
                "save_face_embedding": st.session_state.save_face_embedding,
                "save_face_name": uploaded_name,
                
                "do_detection": st.session_state.do_detection,
                # "which_detector": st.session_state.which_detector,
                "do_recognition": st.session_state.do_recognition,
                # "which_recognizer": st.session_state.which_recognizer
            }
            r = requests.post("http://localhost:8001/settings", json=args)
            # st.write(r)
            
            files = [
                ('files', (uploaded_file.name, bytes_data, uploaded_file.type))
            ]
            response = requests.post("http://localhost:8001/order", files=files)
            result_str = response.json()["products"][0]["result"]
            if result_str == "Success":
                st.write(f"{uploaded_name} 님의 얼굴을 저장 성공했습니다.")
            else:
                st.write("저장 실패했습니다. 다시 시도해주세요.")

        else:
            st.session_state.do_detection = False
            st.session_state.do_recognition = False
