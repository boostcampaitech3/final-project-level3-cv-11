# original code from yolov5-face (https://github.com/deepcam-cn/yolov5-face)
#   - https://github.com/deepcam-cn/yolov5-face/blob/master/detect_face.py


from face_detection.yolov5_face.models.experimental import attempt_load

def load_yolov5(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model
