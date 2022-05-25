from util import Mosaic, DrawRectImg
from args import Args
from PIL import Image

from detection import load_face_db

from retinaface_utils.utils.model_utils import load_model
from retinaface_utils.models.retinaface import RetinaFace
from retinaface_utils.data.config import cfg_mnet
import ml_part as ML

import torch
from facenet_pytorch import MTCNN, InceptionResnetV1 


def init(args):
    model_args = {}
    # 초기에 불러올 모델을 설정하는 공간입니다.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_args['Device'] = device
    # if debug_mode:
    print('Running on device : {}'.format(device))

    # Load mtcnn
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds = [0.6, 0.7, 0.7], factor=0.709, post_process=True, 
        device=device, keep_all=True
    )
    model_args['Mtcnn'] = mtcnn
    
    # Load Detection Model
    if args['DETECTOR'] == 'mtcnn':
        # model_detection = MTCNN(keep_all=True)
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
    face_db = load_face_db("../data/test_images", "./face_db", device)
    model_args['Face_db'] = face_db
    
    return model_args


def ProcessImage(img, args, model_args):
    isPIL = args['IS_PIL']

    # Object Detection
    bboxes = ML.Detection(img, args, model_args)

    # Object Recognition
    face_ids = ML.Recognition(img, bboxes, args, model_args)

    # Mosaic
    img = Mosaic(img, bboxes, face_ids, n=10, isPIL= isPIL)

    # 특정인에 bbox와 name을 보여주고 싶으면
    processed_img = DrawRectImg(img, bboxes, face_ids, isPIL= isPIL)

    return processed_img
    # return img


def main(args):
    # img = Image.open(args['IMAGE_DIR'])
    img = Image.open('../data/dest_images/findobama/twopeople.jpeg')
    # img = cv2.imread('../data/dest_images/findobama/twopeople.jpeg') # CV ver.

    model_args = init(args)

    img = ProcessImage(img, args, model_args)

    img.save(args['SAVE_DIR']+'/output.png', 'png') 
    # cv2.imwrite(args['SAVE_DIR'] + '/output1.jpg', img) # CV ver.

if __name__ == "__main__":
    args = Args().params
    main(args)
