import cv2
import torch

from time import time
from util import Mosaic, DrawRectImg
from args import Args

import ml_part as ML
from database import load_face_db
from facenet_pytorch import InceptionResnetV1

from deep_sort.deep_sort_face import DeepSortFace
from deep_sort.visualization import draw_boxes

from retinaface_utils.utils.model_utils import load_model
from retinaface_utils.models.retinaface import RetinaFace
from retinaface_utils.data.config import cfg_mnet

from detect_face import load_models


def init(args):
    model_args = {}
    # 초기에 불러올 모델을 설정하는 공간입니다.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_args['Device'] = device

    if args['DEBUG_MODE']:
        print('Running on device : {}'.format(device))

    # Load Detection Model
    if args['DETECTOR'] == 'retinaface':
        model_path = 'retinaface_utils/weights/mobilenet0.25_Final.pth'
        backbone_path = './retinaface_utils/weights/mobilenetV1X0.25_pretrain.tar'
        model_detection = RetinaFace(cfg=cfg_mnet, backbone_path=backbone_path, phase = 'test')
        model_detection = load_model(model_detection, model_path, device)
        model_detection.to(device)
        model_detection.eval()
    else: # yolo
        model_detection = load_models("./weights/yolov5n-0.5.pt", device)
        model_detection.to(device)
        model_detection.eval()

    model_args['Detection'] = model_detection

    # Load Recognition Models
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    model_args['Recognition'] = resnet

    model_args['Deepsort'] = DeepSortFace(device=device)

    # Load Face DB
    db_path = "./database"

    face_db = load_face_db("../data_/test_images",
                            db_path,
                            device, args, model_args)

    model_args['Face_db'] = face_db

    return model_args


def ProcessImage(img, args, model_args):
    process_target = args['PROCESS_TARGET']

    # Object Detection
    bboxes, probs = ML.Detection(img, args, model_args)
    if bboxes is None: return img

    # Object Recognition
    face_ids, probs = ML.Recognition(img, bboxes, args, model_args)

    # Mosaic
    img = Mosaic(img, bboxes, face_ids, n=10)

    # 특정인에 bbox와 name을 보여주고 싶으면
    processed_img = DrawRectImg(img, bboxes, face_ids)

    return processed_img


def ProcessVideo(img, args, model_args, id_name):
    # global id_name
    # Object Detection
    bboxes, probs = ML.Detection(img, args, model_args)
    
    # ML.DeepsortRecognition
    
    outputs = ML.Deepsort(img, bboxes, probs, model_args['Deepsort'])
    # last_out = outputs 

    # if boxes is None:
    #     return img, outputs
    
    if len(outputs) > 0:
        bbox_xyxy = outputs[:, :4]
        identities = outputs[:, -1]
        if identities[-1] not in id_name.keys(): # Update가 생기면
            id_name, probs = ML.Recognition(img, bbox_xyxy, args, model_args, id_name, identities)                                       

        img = Mosaic(img, bbox_xyxy, identities, 10, id_name)
    
        # 특정인에 bbox와 name을 보여주고 싶으면
        processed_img = DrawRectImg(img, bbox_xyxy, identities, id_name)
    else:
        processed_img = img
    
    return processed_img, id_name


def main(args):
    model_args = init(args)

    # =================== Image =======================
    image_dir = args['IMAGE_DIR']
    if args['PROCESS_TARGET'] == 'Image':
        # Color channel: BGR
        img = cv2.imread(image_dir)
        img = ProcessImage(img, args, model_args)

        cv2.imwrite(args['SAVE_DIR'] + '/output.jpg', img)
        print('image process complete!')
    # =================== Image =======================

    # =================== Video =======================
    elif args['PROCESS_TARGET'] == 'Video':
        video_path = '../data_/dest_images/son_clip.mp4'

        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        width = int(cap.get(3))
        height = int(cap.get(4))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(args['SAVE_DIR'] + '/output.mp4', fourcc, fps, (width, height))
        # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        id_name = {}
        start = time()
        while True:
            ret, img = cap.read()
            # Color channel: BGR
            if ret:
                if args['TRACKING']:
                    img, id_name = ProcessVideo(img, args, model_args, id_name)
                else:
                    img = ProcessImage(img, args, model_args)
                out.write(img)
            else:
                break

        cap.release()
        out.release()
        print(f'fps: {fps} time: {time() - start} done.')
    # ====================== Video ===========================

    else: # WebCam
        return
        webcam = cv2.VideoCapture(0)
        print('width :%d, height : %d' % (cap.get(3), cap.get(4)))
        
        if not webcam.isOpened():
            print("Could not open webcam")
            exit()

        while webcam.isOpened():
            # read frame from webcam 
            status, frame = webcam.read()

            if not status:
                break

            frame = ProcessImage(frame, args, model_args)
            # display output
            cv2.imshow("Real-time object detection", frame)

            # press "Q" to stop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        # release resources
        webcam.release()
        cv2.destroyAllWindows()   

if __name__ == "__main__":
    args = Args().params
    main(args)
