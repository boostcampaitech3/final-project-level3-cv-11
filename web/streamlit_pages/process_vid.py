import requests
import streamlit as st


def app():
    # 타겟 데이터가 비디오인 경우 활성화 되는 페이지
    st.session_state.save_face_embedding = False
    
    st.text(""); st.text("") # 공백
    st.markdown("###### 처리할 비디오를 업로드")
    
    uploaded_file = st.file_uploader("타겟 비디오를 업로드하세요", type=["mp4"])
    
    if uploaded_file:
        st.text(""); st.text("") # 공백
        st.markdown("###### 데이터 처리 결과")
        
        bytes_data = uploaded_file.getvalue()
        
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
