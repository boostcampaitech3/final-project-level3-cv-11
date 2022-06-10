import cv2
import datetime
import hashlib
import numpy as np
import os
from PIL import Image
import requests
from shutil import rmtree
import streamlit as st


def app(parent_state):
    # 상속
    st.session_state.username = parent_state.username
    st.session_state.do_mosaic = parent_state.do_mosaic
    st.session_state.do_stroke = parent_state.do_stroke
    
    # 타겟 데이터가 이미지인 경우 활성화 되는 페이지
    
    st.text(""); st.text("") # 공백
    st.markdown("###### 처리할 이미지를 업로드")
    
    uploaded_file = st.file_uploader("타겟 이미지를 업로드하세요", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        now = datetime.datetime.now()
        shastr = hashlib.sha224(str(now).encode()).hexdigest()[:8]
        st.session_state.output_dir = f".result_output/{shastr}/"
        
        st.text(""); st.text("") # 공백
        st.markdown("###### 데이터 처리 결과")
        
        uploaded_file_type = uploaded_file.name.split('.')[-1]
        if not os.path.exists(st.session_state.output_dir):
            os.mkdir(st.session_state.output_dir)
        with open(st.session_state.output_dir + f"input.{uploaded_file_type}", "wb") as picfile:
            picfile.write(uploaded_file.getvalue())
        input_file_name = st.session_state.output_dir + f"input.{uploaded_file_type}"
        output_file_name = st.session_state.output_dir + "output.jpg"
        
        args = {
            "USERNAME": st.session_state.username,
            "REQUEST_ID": shastr,
            
            "DO_MOSAIC": st.session_state.do_mosaic,
            "DO_STROKE": st.session_state.do_stroke,

            "INPUT_FILE_NAME": input_file_name,
            "OUTPUT_FILE_NAME": output_file_name
        }
        response = requests.post("http://localhost:8001/order", json=args)
        result_str = response.json()["products"][0]["result"]
        if result_str == "Success":
            col1, col2 = st.columns(2)

            col1.header("Original")
            col1.image(Image.open(input_file_name), use_column_width=True)

            col2.header("Result")
            col2.image(Image.open(output_file_name), use_column_width=True)

            rmtree(st.session_state.output_dir)
