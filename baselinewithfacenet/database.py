import os
import warnings 
import cv2
import numpy as np
import pickle

from torch.utils.data import DataLoader 
from torchvision import datasets
from detection import get_embeddings, recognizer
from retinaface_utils.util import retinaface_detection

warnings.filterwarnings('ignore')
workers = 0 if os.name == 'nt' else 4 


def collate_fn(x):
    return x[0]


def check_face_db(known_images_path, img_db_path):
    # 파일이 있고, 안에 known_images_path 대상이 전부 있는지 파악
    new_image_list = []
    with open(img_db_path, "rb") as load:
        img_db = pickle.load(load)

    image_folder_list = os.listdir(known_images_path)
    for image_folder in image_folder_list:
        if image_folder in img_db:
            # 폴더(이름) 기록이 있으면 내부 파일 검사
            images = os.listdir(os.path.join(known_images_path, image_folder))
            for image in images:
                if image not in img_db[image_folder]:
                    # 이미지 없음, 기존 인물에 새로운 이미지
                    return True
        else:
            # 폴더(이름) 기록이 없음
            return True

    # 폴더, 내부 파일 전부 있음
    return False


def build_db(known_images_path, face_db_path, img_db_path, device, args, model_args):
    assert os.path.exists(known_images_path), 'known_images_path {} not exist'.format(known_images_path)
    
    dataset = datasets.ImageFolder(known_images_path)
    dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

    detector = model_args['Detection'] 
    recognizer = model_args['Recognition']

    face_db = {}

    for x, y in loader : 
        name = dataset.idx_to_class[y]
        
        x_np = np.array(x)
        x = cv2.cvtColor(x_np, cv2.COLOR_RGB2BGR)
        bboxes = retinaface_detection(detector, x, device)
        
        assert bboxes is not None, f'no detection in {name}'
        faces, embedding = get_embeddings(recognizer, x, bboxes, device)
        embedding = embedding.numpy()
        if name in face_db:
            face_db[name].append(embedding)
        else:
            face_db[name] = [embedding]

    with open(face_db_path, "wb") as f:
        pickle.dump(face_db, f)

    img_db = {}
    image_folder_list = os.listdir(known_images_path)
    for image_folder in image_folder_list:
        images = os.listdir(os.path.join(known_images_path, image_folder))
        img_db[image_folder] = images
    with open(img_db_path, "wb") as f:
        pickle.dump(img_db, f)

    print('finished faceDatabase transform!', len(face_db.items()))
    return face_db


def load_face_db(known_images_path, db_path, device, args, model_args):
    db_path = os.path.join(db_path, args['DETECTOR'])
    if not os.path.exists(db_path):
        os.mkdir(db_path)

    face_db_path = os.path.join(db_path, 'face_db')
    img_db_path = os.path.join(db_path, 'img_db')

    if not os.path.exists(face_db_path):
        # 파일 자체가 없으면, 첫 데이터베이스 구축
        print('face_data_path not exist!,try to get face Database transform!')
        face_db = build_db(known_images_path, face_db_path, img_db_path, device, args, model_args)
    else:
        if check_face_db(known_images_path, img_db_path):
            # 그냥 새로 db 구축
            face_db = build_db(known_images_path, face_db_path, img_db_path, device, args, model_args)
        else:
            # 모든 파일 가지고 있음.
            with open(face_db_path, "rb") as f:
                face_db = pickle.load(f)
        
    print('finished load face_data!')
    return face_db
