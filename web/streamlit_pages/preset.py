import streamlit as st


def app():
    # 예시 샘플 페이지
    st.session_state.process_target = None
    
    if "target_type" not in st.session_state:
        st.session_state.target_type = None
    elif st.session_state.target_type not in ("ex1", "ex2", "ex3", "ex4"):
        st.session_state.target_type = None
    
    st.text(""); st.text("") # 공백
    st.markdown("###### 예시 샘플을 선택해주세요")
    
    ex1, ex2, ex3, ex4 = st.columns(4)
    
    button1 = ex1.button("선택", key="ex1")
    ex1.image(".assets/sample_input/examples/example1/example_thumbnail1.jpg")
    
    button2 = ex2.button("선택", key="ex2")
    ex2.image(".assets/sample_input/examples/example2/example_thumbnail2.jpg")
    
    button3 = ex3.button("선택", key="ex3")
    ex3.image(".assets/sample_input/examples/example3/example_thumbnail3.jpg")
    
    button4 = ex4.button("선택", key="ex4")
    ex4.image(".assets/sample_input/examples/example4/example_thumbnail4.jpg")
    
    if button1:
        st.session_state.target_type = "ex1"
    elif button2:
        st.session_state.target_type = "ex2"
    elif button3:
        st.session_state.target_type = "ex3"
    elif button4:
        st.session_state.target_type = "ex4"
    
    
    if "target_type" in st.session_state:
        st.text("") # 공백
        
        if st.session_state.target_type == "ex1":
            st.markdown("**소요 시간** : 00초")
            
            col1, col2 = st.columns(2) # markdown 이전에 있어야 글이 위로 감
            col1.header("Original")
            col1.image(".assets/sample_input/examples/example1/example_input1.png")
            col2.header("Result")
            col2.image(".assets/sample_input/examples/example1/example_output1.png")
            
        elif st.session_state.target_type == "ex2":
            st.markdown("**소요 시간** : 00초")
            
            col1, col2 = st.columns(2) # markdown 이전에 있어야 글이 위로 감
            col1.header("Original")
            col1.image(".assets/sample_input/examples/example2/example_input2.png")
            col2.header("Result")
            col2.image(".assets/sample_input/examples/example2/example_output2.png")
            
        elif st.session_state.target_type == "ex3":
            st.markdown("**소요 시간** : 00초")
            
            col1, col2 = st.columns(2) # markdown 이전에 있어야 글이 위로 감
            col1.header("Original")
            col1.video(".assets/sample_input/examples/example3/example_input3.mp4")
            col2.header("Result")
            col2.video(".assets/sample_input/examples/example3/example_output3.mp4")
            
        elif st.session_state.target_type == "ex4":
            st.markdown("**소요 시간** : 00초")
            
            col1, col2 = st.columns(2) # markdown 이전에 있어야 글이 위로 감
            col1.header("Original")
            col1.video(".assets/sample_input/examples/example4/example_input4.mp4")
            col2.header("Result")
            col2.video(".assets/sample_input/examples/example4/example_output4.mp4")
