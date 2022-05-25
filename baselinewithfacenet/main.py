from matplotlib import image
from util import Mosaic, DrawRectImg
from ml_part import Detection, Recognition
from args import Args
from PIL import Image

from detection import mtcnn_detection, mtcnn_get_embeddings, mtcnn_recognition, load_face_db

from retinaface_utils.util import retinaface_detection
from retinaface_utils.utils.model_utils import load_model
from retinaface_utils.models.retinaface import RetinaFace
from retinaface_utils.data.config import cfg_mnet

import cv2
import torch 
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1 


def ProcessImage(img, bbox_thr, recog_thr, model_detection, mtcnn, resnet, face_db, device, isPIL):
    # Object Detection
    # bboxes = Detection(img, bbox_thr)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if args['DETECTOR'] == 'mtcnn':
        bboxes = mtcnn_detection(model_detection, img, device)
    else:
        bboxes = retinaface_detection(model_detection, img, device)

    print(bboxes)
    faces, unknown_embeddings = mtcnn_get_embeddings(mtcnn, resnet, img, bboxes, device)
    print(faces.shape)

    # Object Recognition
    # unknown_bboxes, face_ids = Recognition(img, bboxes, recog_thr)
    face_ids, result_probs = mtcnn_recognition(img, face_db, unknown_embeddings, recog_thr, device)
    print(face_ids)
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
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

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device : {}'.format(device))

    # Load Detection Model
    if args['DETECTOR'] == 'mtcnn':
        model_detection = MTCNN(keep_all=True)
    else:
        model_path = 'retinaface_utils/weights/mobilenet0.25_Final.pth'
        backbone_path = './retinaface_utils/weights/mobilenetV1X0.25_pretrain.tar'
        model_detection = RetinaFace(cfg=cfg_mnet, backbone_path=backbone_path, phase = 'test')
        model_detection = load_model(model_detection, model_path, device)
        model_detection.to(device)
        model_detection.eval()
    
    # Load Recognition Models
    mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds = [0.6, 0.7, 0.7], factor=0.709, post_process=True, 
    device=device, keep_all=True
    )
    
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # Load Face DB
    face_db = load_face_db("../data/test_images", "./face_db", device)

    img = ProcessImage(img,
                       args['BBOX_THRESHOLD'],
                       args['RECOG_THRESHOLD'],
                       model_detection,
                       mtcnn,
                       resnet,
                       face_db,
                       device,
                       args['IS_PIL']
                       )

    img.save(args['SAVE_DIR']+'/output.png', 'png') 
    # cv2.imwrite(args['SAVE_DIR'] + '/output1.jpg', img) # CV ver.

if __name__ == "__main__":
    args = Args().params
    main(args)