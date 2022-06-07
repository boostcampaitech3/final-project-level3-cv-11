![mofy_banner](.assets/doc/img/MOFY_banner.png)
# MOFY
- Naver boostcamp AI Tech CV-11 CanVas

<br/>

**실행 방법**

```shell
# 1. 백엔드 FastAPI 실행
python backend_app.py

# 2. 프론트엔드 Streamlit 실행 (webcam 환경을 위해 https 필수)
streamlit --log_level "info" run frontend_app.py --server.port 30001 --server.fileWatcherType none
```

<br/>

**프로젝트 구조 📂**

```
canvas-mofy/
│
├── 📂 .assets/
│   ├── 📂 doc/
│   ├── 📂 model_weights/
│   ├── 📂 sample_input/
│   └── 📂 ssl_certificate/
│
├── 📂 .database/
├── 📂 .result_output/
│
├── 📂 configs/
│
├── 📂 face_detection/
│
├── 📂 face_recognition/
│
├── 📂 face_tracking/
│
├── 📂 tests/
│
├── 📂 web/
│   ├── 📂 streamlit_pages/
│   ├── 📂 streamlit_utils/
│   └── 📂 fastapi_utils/
│
├── 📝 frontend_app.py : streamlit을 이용한 사용자 접속용 메인 화면
├── 📝 backend_app.py : fastapi를 이용한 backend
│
├── 📝 .gitignore
├── 📝 README.md
└── 📝 requirements.txt
```

- [.assets](.assets/README.md) : 모델 가중치 파일, 사진 및 영상 파일 등 각종 assets들
  - [doc](.assets/doc/README.md) : 웹 페이지, 쥬피터노트북 파일 또는 README에 활용할 자료들
  - [model_weights](.assets/model_weights/README.md) : 사용할 모델들의 weight 파일들
  - [sample_input](.assets/sample_input/README.md) : 테스트용 사진 및 영상들
  - [ssl_certificate](.assets/ssl_certificate/README.md) : https 연결을 위한 인증서 파일 (보안 상의 이유로 업로드 X)
- [.database](.database/README.md) : 얼굴 임베딩 벡터의 저장 directory
- [.result_output](.result_output/README.md) : 처리 결과 저장 directory
- [configs](configs/README.md)
- [face_detection](face_detection/README.md) : 1단계 - 얼굴 탐지
- [face_recognition](face_recognition/README.md) : 2단계 - 얼굴 인식
- [face_tracking](face_tracking/README.md) : 3단계 - 얼굴 추적
- [tests](tests/README.md) : 각 기능들에 대한 테스트용 .py 스크립트 및 .ipynb 노트북 파일들
- [web](web/README.md) : streamlit 및 fastapi를 이용한 웹 인터페이스 파일들
  - [streamlit_pages](web/streamlit_pages) : frontend 페이지 구성
  - [streamlit_utils](web/streamlit_utils) : frontend 관련 util
  - [fastapi_utils](web/fastapi_utils) : backend 관련 util
- [frontend_app.py](./frontend_app.py) : streamlit을 이용한 사용자 접속용 메인 화면
- [backend_app.py](./backend_app.py) : fastapi를 이용한 backend