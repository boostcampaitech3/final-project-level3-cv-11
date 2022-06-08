import cv2
import numpy as np
from PIL import Image
import requests
import streamlit as st


def app():
    # 타겟 데이터가 이미지인 경우 활성화 되는 페이지
    st.session_state.save_face_embedding = False
    
    st.text(""); st.text("") # 공백
    st.markdown("###### 처리할 이미지를 업로드")
    
    uploaded_file = st.file_uploader("타겟 이미지를 업로드하세요", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        st.text(""); st.text("") # 공백
        st.markdown("###### 데이터 처리 결과")
        
        bytes_data = uploaded_file.getvalue()
        
        files = [
            ('files', (uploaded_file.name, bytes_data, uploaded_file.type))
        ]
        response = requests.post("http://localhost:8001/order", files=files)
        result_str = response.json()["products"][0]["result"]
        st.write(result_str) # Success
        
        col1, col2 = st.columns(2)
        
        col1.header("Original")
        encoded_img = np.fromstring(bytes_data, dtype=np.uint8)
        img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
        col1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)
        
        col2.header("Result")
        col2.image(Image.open(".result_output/output.png"), use_column_width=True)
