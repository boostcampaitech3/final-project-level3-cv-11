import datetime
import hashlib
import requests
from shutil import rmtree
import streamlit as st


def app():
    # 타겟 데이터가 비디오인 경우 활성화 되는 페이지
    
    st.text(""); st.text("") # 공백
    st.markdown("###### 처리할 비디오를 업로드")
    
    uploaded_file = st.file_uploader("타겟 비디오를 업로드하세요", type=["mp4"])
    
    if uploaded_file:
        now = datetime.datetime.now()
        shastr = hashlib.sha224(str(now).encode()).hexdigest()[:8]
        st.session_state.output_dir = f".result_output/{shastr}/"
        
        st.text(""); st.text("") # 공백
        st.markdown("###### 데이터 처리 결과")
        
        if not os.path.exists(st.session_state.output_dir):
            os.mkdir(st.session_state.output_dir)
        with open(st.session_state.output_dir + "input.mp4", "wb") as vidfile:
            vidfile.write(uploaded_file.getvalue())
        input_file_name = st.session_state.output_dir + "input.mp4"
        output_file_name = st.session_state.output_dir + "output.mp4"
        
        args = {
            "USERNAME": st.session_state.username,
            "REQUEST_ID": shastr,

            "INPUT_FILE_NAME": input_file_name,
            "OUTPUT_FILE_NAME": output_file_name,

            "SAVE_FACE_NAME": uploaded_name
        }
        response = requests.post("http://localhost:8001/order", json=args)
        result_str = response.json()["products"][0]["result"]
        if result_str == "Success":
            col1, col2 = st.columns(2)

            col1.header("Original")
            col1.video(input_file_name)

            col2.header("Result")
            col2.video(output_file_name)

            rmtree(st.session_state.output_dir)
