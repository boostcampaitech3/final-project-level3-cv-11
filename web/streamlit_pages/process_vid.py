import requests
import streamlit as st


def app():
    # 타겟 데이터가 비디오인 경우 활성화 되는 페이지
    st.session_state.process_target = "vid"
    st.session_state.save_face_embedding = False
    
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
    # st.session_state.which_tracker = choice_col3.selectbox(
    #     "Tracking algorithm", 
    #     ("DeepSort", ), 
    #     index=0
    # )
    
    st.text(""); st.text("") # 공백
    st.markdown("###### 처리할 비디오를 업로드")
    
    uploaded_file = st.file_uploader("타겟 비디오를 업로드하세요", type=["mp4"])
    
    if uploaded_file:
        st.session_state.do_detection = True
        st.session_state.do_recognition = True
        st.session_state.do_tracking = True
        
        st.text(""); st.text("") # 공백
        st.markdown("###### 데이터 처리 결과")
        
        bytes_data = uploaded_file.getvalue()
        
        args = {
            "PROCESS_TARGET": st.session_state.process_target,
            "SAVE_FACE_EMBEDDING": st.session_state.save_face_embedding,
            
            "DO_DETECTION": st.session_state.do_detection,
            # "WHICH_DETECTOR": st.session_state.which_detector,
            "DO_RECOGNITION": st.session_state.do_recognition,
            # "WHICH_RECOGNIZER": st.session_state.which_recognizer,
            "DO_TRACKING": st.session_state.do_tracking,
            # "WHICH_TRACKER": st.session_state.which_tracker
        }
        r = requests.post("http://localhost:8001/settings", json=args)
        # st.write(r)
        
        files = [
            ('files', (uploaded_file.name, bytes_data, uploaded_file.type))
        ]
        response = requests.post("http://localhost:8001/order", data=args, files=files)
        result_str = response.json()["products"][0]["result"]
        st.write(result_str) # Success
        
        col1, col2 = st.columns(2)
        
        col1.header("Original")
        with open(".result_output/input_video.mp4", "wb") as fp:
            fp.write(bytes_data)
        col1.video(".result_output/input_video.mp4")
        
        col2.header("Result")
        col2.video(".result_output/output_video.mp4")

    else:
        st.session_state.do_detection = False
        st.session_state.do_recognition = False
        st.session_state.do_tracking = False
