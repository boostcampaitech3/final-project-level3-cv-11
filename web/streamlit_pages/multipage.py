# original code by Prakhar Rathi (https://github.com/prakharrathi25)
#   - https://github.com/prakharrathi25/data-storyteller/blob/main/multipage.py
#
# This file is the framework for generating multiple Streamlit applications 
# through an object oriented framework. 


# Import necessary libraries
import requests
import streamlit as st


# Define the multipage class to manage the multiple apps in our program 
class MultiPage: 
    """Framework for combining multiple streamlit applications."""

    def __init__(self) -> None:
        """Constructor class to generate a list which will store all our applications as an instance variable."""
        self.pages = []
        
        self.prev_state = {
            "process_target": None,
            
            "which_detector": "YOLOv5",
            "which_recognizer": "FaceNet",
            "save_face_embedding": False,
            "which_tracker": "DeepSort",
            
            "do_mosaic": True,
            "do_stroke": False,
            
            "debug_mode": False
        }
        st.session_state.process_target = None
        st.session_state.do_detection = False
        st.session_state.do_recognition = False
        st.session_state.do_tracking = False
    
    def add_page(self, title, func) -> None: 
        """Class Method to Add pages to the project
        
        Args:
            title ([str]): The title of page which we are adding to the list of apps 
            func: Python function to render this page in Streamlit
        """

        self.pages.append({
                "title": title, 
                "function": func
            })

    def run(self, parent_state):
        st.session_state.username = parent_state.username # 상속
        
        # Drodown to select the page to run
        with st.sidebar:
            page = st.selectbox(
                "현재 페이지", 
                self.pages, 
                format_func=lambda page: page['title']
            )
            
            st.text(""); st.text("") # 공백
            col1, col2 = st.columns(2)
            st.session_state.do_mosaic = col1.checkbox("모자이크", value=True)
            st.session_state.do_stroke = col2.checkbox("스트로크", value=False)
            
            st.text(""); st.text("") # 공백
            st.session_state.which_detector = st.selectbox(
                "탐지 알고리즘 선택", 
                ("RetinaFace", "YOLOv5"), # "MTCNN", "FaceBoxes", "HaarCascades"
                index=1
            )
            
            st.text("") # 공백
            st.session_state.which_recognizer = st.selectbox(
                "인식 알고리즘 선택", 
                ("FaceNet", ), # "ArcFace", "ArcFace_Mofy"
                index=0
            )
            
            st.text("") # 공백
            st.session_state.which_tracker = st.selectbox(
                "추적 알고리즘 선택", 
                ("DeepSort", ), 
                index=0
            )
            
            st.text(""); st.text("") # 공백
            st.session_state.debug_mode = st.checkbox("디버그 모드")
            
            if page["title"] == "샘플 페이지":
                st.session_state.process_target = None
            elif page["title"] in ("얼굴 정보 등록", "이미지 모자이크 처리"):
                st.session_state.process_target = "img"
            elif page["title"] in ("비디오 모자이크 처리", "웹캠 모자이크 처리"):
                st.session_state.process_target = "vid"
            else:
                raise ValueError(page["title"])
            
            self.observe_session_state_change()
            if st.session_state.debug_mode:
                debug_string = "Current mode: "
                if st.session_state.do_detection:
                    debug_string += st.session_state.which_detector
                if st.session_state.do_recognition:
                    debug_string += " / " + st.session_state.which_recognizer
                if st.session_state.do_tracking:
                    debug_string += " / " + st.session_state.which_tracker
                st.text(debug_string)

        # run the app function 
        page['function'](st.session_state)
    
    def observe_session_state_change(self):
        send_request = False
        
        if self.prev_state["process_target"] != st.session_state.process_target:
            self.prev_state["process_target"] = st.session_state.process_target; send_request = True
            if st.session_state.process_target is None:
                st.session_state.do_detection = False
                st.session_state.do_recognition = False
                st.session_state.do_tracking = False
            elif st.session_state.process_target == "img":
                st.session_state.do_detection = True
                st.session_state.do_recognition = True
                st.session_state.do_tracking = False
            elif st.session_state.process_target == "vid":
                st.session_state.do_detection = True
                st.session_state.do_recognition = True
                st.session_state.do_tracking = True
            else:
                raise NotImplementedError(st.session_state.process_target)
        
        if self.prev_state["which_detector"] != st.session_state.which_detector:
            self.prev_state["which_detector"] = st.session_state.which_detector; send_request = True
        
        if self.prev_state["which_recognizer"] != st.session_state.which_recognizer:
            self.prev_state["which_recognizer"] = st.session_state.which_recognizer; send_request = True
        
        if self.prev_state["which_tracker"] != st.session_state.which_tracker:
            self.prev_state["which_tracker"] = st.session_state.which_tracker; send_request = True
        
        if self.prev_state["do_mosaic"] != st.session_state.do_mosaic:
            self.prev_state["do_mosaic"] = st.session_state.do_mosaic
        
        if self.prev_state["do_stroke"] != st.session_state.do_stroke:
            self.prev_state["do_stroke"] = st.session_state.do_stroke
        
        if self.prev_state["debug_mode"] != st.session_state.debug_mode:
            self.prev_state["debug_mode"] = st.session_state.debug_mode
        
        if send_request:
            args = {
                "PROCESS_TARGET": st.session_state.process_target,
                
                "USERNAME": st.session_state.username,

                "DO_DETECTION": st.session_state.do_detection,
                "WHICH_DETECTOR": st.session_state.which_detector,
                "DO_RECOGNITION": st.session_state.do_recognition,
                "WHICH_RECOGNIZER": st.session_state.which_recognizer,
                "DO_TRACKING": st.session_state.do_tracking,
                "WHICH_TRACKER": st.session_state.which_tracker
            }
            r = requests.post("http://localhost:8001/settings", json=args)
            # st.write(r)
        