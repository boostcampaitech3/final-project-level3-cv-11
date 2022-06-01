from time import time
import cv2
import torch
import torchvision

from util import Mosaic, DrawRectImg, TrackingID
from args import Args

import ml_part as ML
from detection import load_face_db
from facenet_pytorch import MTCNN, InceptionResnetV1

from retinaface_utils.utils.model_utils import load_model
from retinaface_utils.models.retinaface import RetinaFace
from retinaface_utils.data.config import cfg_mnet


def init(args):
    model_args = {}
    # 초기에 불러올 모델을 설정하는 공간입니다.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_args['Device'] = device
    if args['DEBUG_MODE']:
        print('Running on device : {}'.format(device))

    # Load mtcnn
    # 이걸로 image crop하는데 이 것도 나중에 자체 기능으로 빼야함. util.py의 CropRoiImg를 좀 쓰면 될 듯.
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds = [0.6, 0.7, 0.7], factor=0.709, post_process=True, 
        device=device, keep_all=True
    )
    model_args['Mtcnn'] = mtcnn
    
    # Load Detection Model
    if args['DETECTOR'] == 'mtcnn':
        # model_detection = MTCNN(keep_all=True)
        # 나중에 image crop 자체 기능 구현되면 위 코드를 여기로
        model_detection = mtcnn
    else:
        model_path = 'retinaface_utils/weights/mobilenet0.25_Final.pth'
        backbone_path = './retinaface_utils/weights/mobilenetV1X0.25_pretrain.tar'
        model_detection = RetinaFace(cfg=cfg_mnet, backbone_path=backbone_path, phase = 'test')
        model_detection = load_model(model_detection, model_path, device)
        model_detection.to(device)
        model_detection.eval()
    model_args['Detection'] = model_detection

    # Load Recognition Models
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    model_args['Recognition'] = resnet

    # Load Face DB
    if args['DETECTOR'] == 'retinaface':
        face_db = load_face_db("../data/test_images", "./face_db_BGR", device, args)
    else:
        face_db = load_face_db("../data/test_images", "./face_db", device, args)
    model_args['Face_db'] = face_db
    
    return model_args


def ProcessImage(img, args, model_args, known_ids = None):
    process_target = args['PROCESS_TARGET']

    # Object Detection
    bboxes = ML.Detection(img, args, model_args)
    if bboxes is None:
        if args['DETECTOR'] == 'mtcnn':
            if process_target == 'Video': # torchvision
                img = img.numpy()
            # Color channel: RGB -> BGR
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    if args['DETECTOR'] == 'mtcnn':
        # 모자이크 전처리
        if process_target == 'Video': # torchvision
            img = img.numpy()
        # Color channel: RGB -> BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Object Recognition
    face_ids = ML.Recognition(img, bboxes, args, model_args, known_ids)
    # 이번 프레임의 ids 수집
    if known_ids is not None:
        known_ids.get_ids(face_ids, bboxes)

    
    # Mosaic
    img = Mosaic(img, bboxes, face_ids, n=10)

    # 특정인에 bbox와 name을 보여주고 싶으면
    # 임시 카운트
    processed_img = DrawRectImg(img, bboxes, face_ids, known_ids)

    return (processed_img)


def main(args):
    model_args = init(args)
    # =================== Image =======================
    if args['PROCESS_TARGET'] == 'Image':
        # Color channel: BGR
        img = cv2.imread(args['IMAGE_DIR'])
        if args['DETECTOR'] == 'mtcnn':
            # Color channel: BGR -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = ProcessImage(img, args, model_args)

        cv2.imwrite(args['SAVE_DIR'] + '/output.jpg', img)
    # =================== Image =======================

    # =================== Video =======================
    elif args['PROCESS_TARGET'] == 'Video':
        video_path = '../data/dest_images/kakao/song.mp4'
        #video_path = '../paddlevideo/mp4s/test.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(args['SAVE_DIR'] + '/output.mp4', fourcc, 24.0, (1280,720))
        known_ids = TrackingID(IoU_thr=.8, IoU_weight=.4)

        start = time()
        if args['DETECTOR'] == 'retinaface':
            cap = cv2.VideoCapture(video_path)
            while True:
                ret, img = cap.read()
                # Color channel: BGR
                if ret:
                    img = ProcessImage(img, args, model_args, known_ids)

                    out.write(img)
                    #print('done')
                else:
                    break
            cap.release()

        elif args['DETECTOR'] == 'mtcnn':
            video = torchvision.io.VideoReader(video_path, stream = 'video')
            # 나중에 video resolution, fps 바꾸는 옵션 넣어야 함. video.get_metadata() 이용하면 될 듯.
            if args['DEBUG_MODE']:
                print(video.get_metadata())
            video.set_current_stream('video')

            for frame in video:
                img = frame['data']
                # img.to(model_args['Device'])
                # Color channel: RGB
                img = torch.permute(img, (1, 2, 0))
                img = ProcessImage(img, args, model_args, known_ids)

                out.write(img)

        out.release()
        print('done.', time() - start)
    # ====================== Video ===========================


if __name__ == "__main__":
    args = Args().params
    main(args)
