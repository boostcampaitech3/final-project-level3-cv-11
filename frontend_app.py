# original code by Prakhar Rathi (https://github.com/prakharrathi25)
#   - https://github.com/prakharrathi25/data-storyteller/blob/main/app.py

import io
import numpy as np
import os

from datetime import datetime
from PIL import Image
from time import sleep

import streamlit as st

# Custom imports
from web.streamlit_utils.confirm_button_hack import cache_on_button_press
import web.streamlit_utils.reload; import importlib

from web.streamlit_pages import preset, register_face, process_img, process_vid, process_cam # import your pages here
from web.streamlit_pages.multipage import MultiPage


root_password = "password"
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False # 비밀번호 기능을 활성화 하려면 False


favicon = Image.open(".assets/doc/ico/MOFY_32x32.png")
st.set_page_config(page_title="MOFY: MOsaic For You", page_icon=favicon, layout="wide", initial_sidebar_state="auto")

# Create an instance of the app 
app = MultiPage()

# Add all your application here
app.add_page("샘플 페이지", preset.app)
app.add_page("얼굴 정보 등록", register_face.app)
app.add_page("이미지 모자이크 처리", process_img.app)
app.add_page("비디오 모자이크 처리", process_vid.app)
app.add_page("웹캠 모자이크 처리", process_cam.app)


if not st.session_state.authenticated:
    # 로그인 화면
    st.image(".assets/doc/img/MOFY_banner.png")
    
    @cache_on_button_press("비밀번호 입력")
    def authenticate(password) -> bool:
        if st.session_state.username == "root":
            return password == root_password
        else:
            return True # TODO: 현재 root 이외의 모든 유저는 무조건 통과

    st.session_state.username = st.text_input("회원님의 이름을 입력하세요.", value="guest")
    password = st.text_input("비밀번호를 입력하세요.", type="password")
    if authenticate(password):
        st.success('You are authenticated!')
        st.session_state.authenticated = True
        
        # st.text("3초 후 페이지가 새로고침 됩니다.")
        
        # sleep(3)
        # with open("web/streamlit_utils/reload.py", 'w') as dummy_script:
        #     # 코드 변경 시 streamlit은 자동으로 재실행
        #     #   - TODO: 다른 사람이 로그인해서 코드 변경 시 페이지가 재실행 되는 상황이 발생할 수도?
        #     dummy_script.write(f"access_datetime = '{datetime.now().strftime('%y%m%d %H%M%S')}'")
        # try:
        #     importlib.reload(web.streamlit_utils.reload)
        # except:
        #     pass
        st.button("바로 접속") # 이상하게 3초 후에 무작위로 새로고침이 안되는 현상이 있음. 빈 버튼과 상호작용 시 새로고침 됨.
        
    else:
        st.error('The password is invalid.')

else: # if st.session_state.authenticated:
    # Title of the main page
    col1, col2 = st.columns(2)
    col1.image(".assets/doc/img/MOFY_logo.png")
    col2.markdown(f'<p align="right">안녕하세요, <b>{st.session_state.username}</b>님!</p>', unsafe_allow_html=True)
    st.title("MOFY: MOsaic For You")
    
    # The main app
    app.run(st.session_state)
