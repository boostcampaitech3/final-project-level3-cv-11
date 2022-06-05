from tkinter import X
import torch
import os
import warnings 
import cv2
import numpy as np
import pickle

import ml_part as ML
from PIL import Image
from torch.utils.data import DataLoader 
from torchvision import datasets
from detection import get_embeddings

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
                    new_image_list.append([image_folder,
                        os.path.join(known_images_path, image_folder, image)])
            img_db[image_folder] = images
        else:
            # 폴더(이름) 기록이 없음, 새로운 인물
            images = os.listdir(os.path.join(known_images_path, image_folder))
            for image in images:
                new_image_list.append([image_folder,
                    os.path.join(known_images_path, image_folder, image)])
            img_db[image_folder] = images
    
    if not new_image_list:
        # 추가된 데이터가 있다면
        with open(img_db_path, "wb") as save:
            pickle.dump(img_db, save)

    # [[name, path], ...]
    return new_image_list


def get_embedding(face_db_path, new_img_list, mtcnn, resnet, device, args):
    with open(face_db_path, "rb") as f:
        face_db = pickle.load(f)
    aligned = []
    names = []

    for name, path in new_img_list : 
        if args['DETECTOR'] == 'retinaface':
            img = cv2.imread(path)
        else:
            if path[-3:].upper() == 'PNG':
                img = Image.open(path).convert('RGB')
            else:
                img = Image.open(path)
        # 자체 기능이 바로 crop까지 해줘서 쓰는데 나중에 바꿔야함
        x_aligned, probs = mtcnn(img, return_prob=True)
        if x_aligned is not None:
            print('Face detected with probability : {:8f}'.format(probs[0]))
            aligned.append(x_aligned[0])
            names.append(name)
    aligned = torch.stack(aligned).to(device)
    embeddings = resnet(aligned).detach().cpu()

    for i, eb in enumerate(embeddings): 
        if names[i] not in face_db:
            face_db[names[i]] = [eb]
        else:
            face_db[names[i]].append(eb)
    
    return face_db


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
        
<<<<<<< HEAD
        x_np = np.array(x)
        x = cv2.cvtColor(x_np, cv2.COLOR_RGB2BGR) 
        bboxes, probs = ML.Detection(x, args, model_args)
=======
        if args['DETECTOR'] == 'retinaface':
            x_np = np.array(x)
            x = cv2.cvtColor(x_np, cv2.COLOR_RGB2BGR)
            bboxes = retinaface_detection(detector, x, device)
        elif args['DETECTOR'] == 'yolo':
            x_np = np.array(x)
            x = cv2.cvtColor(x_np, cv2.COLOR_RGB2BGR)
            bboxes, probs = yolo_detection(detector, x, device)
>>>>>>> eb528c8901f19d461b90132bf998d270d1f8b793

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
        # 파일 자체가 없으면, 첫 기동
        print('face_data_path not exist!,try to get face Database transform!')
        face_db = build_db(known_images_path, face_db_path, img_db_path, device, args, model_args)
        return face_db

    else:
        new_img_list = check_face_db(known_images_path, img_db_path)
        if new_img_list:
            # 그냥 새로 db 구축
            face_db = build_db(known_images_path, face_db_path, img_db_path, device, args, model_args)
            # 일부 파일 추가 필요
            # print(len(new_img_list), 'files need get embedding! please wait...')
            # mtcnn = model_args['Mtcnn']
            # resnet = model_args['Recognition']
            # face_db = get_embedding(face_db_path, new_img_list, mtcnn, resnet, device, args)
            # with open(face_db_path, "wb") as f:
            #     pickle.dump(face_db, f)
        else:
            # 모든 파일 가지고 있음.
            with open(face_db_path, "rb") as f:
                face_db = pickle.load(f)
        
    print('finished load face_data!')
    return face_db
