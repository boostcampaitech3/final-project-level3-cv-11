from matplotlib import image
from util import Mosaic, DrawRectImg
from ml_part import Detection, Recognition
from args import Args
from PIL import Image

from detection import mtcnn_detection, mtcnn_get_embeddings, mtcnn_recognition
import torch 


def ProcessImage(img, bbox_thr, recog_thr, device):
    # Object Detection
    # bboxes = Detection(img, bbox_thr)
    bboxes, landmarks = mtcnn_detection(img, bbox_thr, device)
    print(bboxes)
    faces, unknown_embeddings = mtcnn_get_embeddings(img, bboxes, device)

    # Object Recognition
    # unknown_bboxes, face_ids = Recognition(img, bboxes, recog_thr)
    face_ids, result_probs = mtcnn_recognition(img, unknown_embeddings, recog_thr, device)
    print(face_ids)

    # Mosaic
    img = Mosaic(img, bboxes, face_ids, n=10, isPIL= True) 

    # 특정인에 bbox와 name을 보여주고 싶으면
    processed_img = DrawRectImg(img, bboxes, landmarks, face_ids)

    return processed_img
    # return img


def main(args):
    img = Image.open(args['IMAGE_DIR'])
    # img = Image.open('../data/dest_images/kakao3.jpeg')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device : {}'.format(device))

    img = ProcessImage(img,
                       args['BBOX_THRESHOLD'],
                       args['RECOG_THRESHOLD'], device)

    img.save(args['SAVE_DIR']+'/output.png', 'png')


if __name__ == "__main__":
    args = Args().params
    main(args)