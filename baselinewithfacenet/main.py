from matplotlib import image
from util import Mosaic, DrawRectImg
from ml_part import Detection, Recognition
from args import Args
from PIL import Image

from detection import mtcnn_detection, mtcnn_get_embeddings, mtcnn_recognition
import torch 

import cv2


def ProcessImage(img, bbox_thr, recog_thr, device, isPIL):
    # Object Detection
    # bboxes = Detection(img, bbox_thr)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bboxes = mtcnn_detection(img, bbox_thr, device)
    print(bboxes)
    faces, unknown_embeddings = mtcnn_get_embeddings(img, bboxes, device)

    # Object Recognition
    # unknown_bboxes, face_ids = Recognition(img, bboxes, recog_thr)
    face_ids, result_probs = mtcnn_recognition(img, unknown_embeddings, recog_thr, device)
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
    img = Image.open('../data/dest_images/kakao/Kakao2.jpg')
    # img = cv2.imread('../data/dest_images/kakao/Kakao2.jpg') # CV ver.

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device : {}'.format(device))

    img = ProcessImage(img,
                       args['BBOX_THRESHOLD'],
                       args['RECOG_THRESHOLD'],
                       device,
                       args['IS_PIL']
                       )

    img.save(args['SAVE_DIR']+'/output.png', 'png') 
    # cv2.imwrite(args['SAVE_DIR'] + '/output1.jpg', img) # CV ver.

if __name__ == "__main__":
    args = Args().params
    main(args)