from PIL import ImageDraw
from tqdm import tqdm
import os, cv2
import numpy as np
from PIL import ImageDraw, ImageFont, Image
import pickle

def show_results(img, bounding_boxes, facial_landmarks = []):
    """Draw bounding boxes and facial landmarks.
    Arguments:
        img: an instance of PIL.Image.
        bounding_boxes: a float numpy array of shape [n, 5].
        facial_landmarks: a float numpy array of shape [n, 10].
    Returns:
        an instance of PIL.Image.
    """
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)

    for b in bounding_boxes:
        draw.rectangle([
            (b[0], b[1]), (b[2], b[3])
        ], outline = 'white')

    inx = 0
    for p in facial_landmarks:
        for i in range(5):
            draw.ellipse([
                (p[i] - 1.0, p[i + 5] - 1.0),
                (p[i] + 1.0, p[i + 5] + 1.0)
            ], outline = 'blue')

    return img_copy


def add_text(img, text, left, top, color=(0, 0, 0), size=20):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        # font = ImageFont.truetype('simsun.ttc', size)
        font = cv2.FONT_HERSHEY_SIMPLEX 
        draw.text((left, top), text, color,font=font)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def draw_face(img, boxes_c, facial_landmarks = []):
        if boxes_c is not None:
            for i in range(boxes_c.shape[0]):
                bbox = boxes_c[i, :4]
                corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                cv2.rectangle(img, (corpbbox[0], corpbbox[1]),
                              (corpbbox[2], corpbbox[3]), (255, 0, 0), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX  
                # img = add_text(img, 'unknown', corpbbox[0], corpbbox[1] + 25, color=(255, 255, 0), size=30)
                img = cv2.putText(img, "unknown", (corpbbox[0], corpbbox[1]), font, 0.5, (0, 255, 0), 1)

            # # Landmarks 
            # inx = 0
            # for p in facial_landmarks:
            #     for i in range(5):
            #         cv2.circle(img, 
            #             (int(p[i] - 1.0), int(p[i + 5] - 1.0)),
            #             radius=3, color=(255,0,0), thickness=-1)
        return img