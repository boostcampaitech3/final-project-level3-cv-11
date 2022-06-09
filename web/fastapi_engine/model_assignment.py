import torch
import torch.backends.cudnn as cudnn

# ----- Detection ----- #
from face_detection.mtcnn import MTCNN

from face_detection.retinaface import RetinaFace
from face_detection.retinaface.utils.load_util import load_model as load_retinanet
from configs.model.retinaface_config import cfg_mnet

from face_detection.yolov5_face import load_yolov5

from face_detection.faceboxes import FaceBoxes
from face_detection.faceboxes.utils.load_util import load_model as load_faceboxes

import cv2
from cv2 import CascadeClassifier

# ----- Recognition ----- #
from face_recognition.facenet import InceptionResnetV1

# ----- Tracking ----- #
from face_tracking.deepsort.deep_sort_face import DeepSortFace


def assign_detector(which_detector, device):
    assert which_detector in ("MTCNN", "RetinaFace", "YOLOv5", "FaceBoxes", "HaarCascades"), which_detector
    
    if which_detector == "MTCNN":
        model = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds = [0.6, 0.7, 0.7], factor=0.709, post_process=True, 
            device=device, keep_all=True
        ).eval()
        raise NotImplementedError(which_detector)
    elif which_detector == "RetinaFace":
        model_path = ".assets/model_weights/detector/retinaface/mobilenet0.25_Final.pth"
        backbone_path = ".assets/model_weights/detector/retinaface/mobilenetV1X0.25_pretrain.tar"
        model = RetinaFace(cfg=cfg_mnet, backbone_path=backbone_path, phase="test")
        model = load_retinanet(model, model_path, device)
        model.to(device)
        model.eval()
    elif which_detector == "YOLOv5":
        model_path = ".assets/model_weights/detector/yolov5_face/yolov5-blazeface.pt"
        model = load_yolov5(model_path, device)
        model.to(device)
        model.eval()
    elif which_detector == "FaceBoxes":
        torch.set_grad_enabled(False)
        model_path = ".assets/model_weights/detector/faceboxes/FaceBoxesProd.pth"
        model = FaceBoxes(phase='test', size=None, num_classes=2)    # initialize detector
        model = load_faceboxes(model, model_path, load_to_cpu=True)
        model.eval()
        cudnn.benchmark = True
        device = "cpu" # torch.device("cpu" if args.cpu else "cuda")
        model = model.to(device)
        raise NotImplementedError(which_detector)
    else: # if which_detector == "HaarCascades":
        xml_path = ".assets/model_weights/detector/haarcascades/haarcascade_frontalface_default.xml"
        model = CascadeClassifier(cv2.samples.findFile(xml_path))
        raise NotImplementedError(which_detector)
    
    return model


def assign_recognizer(which_recognizer, device):
    assert which_recognizer in ("FaceNet", "ArcFace", "ArcFace_Mofy"), which_recognizer
    
    if which_recognizer == "FaceNet":
        model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    elif which_recognizer == "ArcFace":
        model = None
        raise NotImplementedError(which_recognizer)
    else: # if which_recognizer == "ArcFace_Mofy":
        model = None
        raise NotImplementedError(which_recognizer)
    
    return model


def assign_tracker(which_tracker, device):
    assert which_tracker in ("DeepSort", ), which_tracker
    
    if which_tracker == "DeepSort":
        algo = DeepSortFace(device=device)
    else:
        algo = None
        raise NotImplementedError(which_tracker)
    
    return algo
