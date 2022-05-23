from matplotlib import image
from util import Mosaic, DrawRectImg
from ml_part import Detection, Recognition
from args import Args
from PIL import Image


def ProcessImage(img, bbox_thr, recog_thr):
    # Object Detection
    bboxes = Detection(img, bbox_thr)

    # Object Recognition
    unknown_bboxes, face_ids = Recognition(img, bboxes, recog_thr)

    mosaiced_img = Mosaic(img, unknown_bboxes, 3)
    # 특정인에 bbox와 name을 보여주고 싶으면
    processed_img = DrawRectImg(mosaiced_img, face_ids)

    return processed_img


def main(args):
    img = Image.open(args['IMAGE_DIR'])

    img = ProcessImage(img,
                       args['BBOX_THRESHOLD'],
                       args['RECOG_THRESHOLD'])

    img.save(args['SAVE_DIR']+'/ouput.png', 'png')


if __name__ == "__main__":
    args = Args().params
    main(args)