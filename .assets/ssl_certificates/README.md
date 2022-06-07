### streamlit을 강제로 https로 서비스 되도록

1. OpenSSL로 Self-Signed Certificates 생성 (참고: [링크](https://mariadb.com/docs/security/data-in-transit-encryption/create-self-signed-certificates-keys-openssl/))

   ```bash
   (sudo) apt install openssl
   
   # 1. Generate a private key for the Certificate Authority:
   openssl genrsa 2048 > ca-key.pem
   
   # 2. Generate the X509 certificate for the Certificate Authority:
   openssl req -new -x509 -nodes -days 365 -key ca-key.pem -out ca-cert.pem
   
   # 3. 원하는 위치로 해당 파일을 이동해주세요
   mv ca-cert.pem ./.assets/ssl_certificates/
   mv ca-key.pem ./.assets/ssl_certificates/
   ```

2. 가상환경을 생성한 후 venv 안의 streamlit 파이썬 코드를 변경 (참고: [링크](https://discuss.streamlit.io/t/how-to-run-streamlit-in-https-mode/18426/7))

   - `venv/lib/python3.8/site-packages/streamlit/server/server.py`의 [start_listening()](https://github.com/streamlit/streamlit/blob/develop/lib/streamlit/server/server.py#L176-L178) 메소드

   ```shell
   http_server = HTTPServer(
       app, max_buffer_size=config.get_option("server.maxUploadSize") * 1024 * 1024,
       ssl_options={
           "certfile": "/opt/ml/canvas-mofy/.assets/ssl_certificates/ca-cert.pem",
           "keyfile": "/opt/ml/canvas-mofy/.assets/ssl_certificates/ca-key.pem"
       }
   )
   ```

3. protobuf 패키지를 3.20.x 또는 이하의 버젼으로 다운그레이드 (다운그레이드 하지 않으면 아래와 같은 에러가 발생한다.)

   ```bash
   Traceback (most recent call last):
     File "/opt/ml/final-project-level3-cv-11/venv/bin/streamlit", line 5, in <module>
       from streamlit.cli import main
     File "/opt/ml/final-project-level3-cv-11/venv/lib/python3.8/site-packages/streamlit/__init__.py", line 48, in <module>
       from streamlit.proto.RootContainer_pb2 import RootContainer
     File "/opt/ml/final-project-level3-cv-11/venv/lib/python3.8/site-packages/streamlit/proto/RootContainer_pb2.py", line 33, in <module>
       _descriptor.EnumValueDescriptor(
     File "/opt/ml/final-project-level3-cv-11/venv/lib/python3.8/site-packages/google/protobuf/descriptor.py", line 755, in __new__
       _message.Message._CheckCalledFromGeneratedFile()
   TypeError: Descriptors cannot not be created directly.
   If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0.
   If you cannot immediately regenerate your protos, some other possible workarounds are:
    1. Downgrade the protobuf package to 3.20.x or lower.
    2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python parsing and will be much slower).
   
   More information: <https://developers.google.com/protocol-buffers/docs/news/2022-05-06#python-updates>
   ```

   - `pip install protobuf==3.20.1`

4. `http://<서버IP>:<streamlit 포트>` 가 아니라 `https://<서버IP>:<streamlit 포트>`로 접속

- 한계점: 이 방식으로는 streamlit cloud 서비스를 이용할 수 없음. 환경을 제어 가능한 aistages 서버에서만 가능할 것으로 보임