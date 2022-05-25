from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face 
from PIL import Image, ImageDraw
import torch 
from torch.utils.data import DataLoader 
from torchvision import datasets 
import numpy as np 
import pandas as pd 
import os
from tqdm import tqdm
import warnings 
import pickle
warnings.filterwarnings('ignore')
workers = 0 if os.name == 'nt' else 4 

def collate_fn(x):
    return x[0]

def mtcnn_detection(model, img, device):
    bboxes, probs = model.detect(img, landmarks=False)
    return bboxes

def mtcnn_get_embeddings(mtcnn, resnet, img, bboxes, device, save_path=None):
    faces = mtcnn.extract(img, bboxes, save_path)
    # print(faces.shape)
    faces = faces.to(device)
    unknown_embeddings = resnet(faces).detach().cpu()
    return faces, unknown_embeddings


def update_face_db(known_images_path, device):
    face_db = {}
    face_db_path = "./face_db"
    assert os.path.exists(known_images_path), 'known_images_path {} not exist'.format(known_images_path)
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds = [0.6, 0.7, 0.7], factor=0.709, post_process=True, 
        device=device, keep_all=True
    )
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    dataset = datasets.ImageFolder(known_images_path)
    dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)
    aligned = []
    names = []
    face_db = {}
    for x, y in loader : 
        x_aligned, probs = mtcnn(x, return_prob=True)
        if x_aligned is not None: 
            print('Face detected with probability : {:8f}'.format(probs[0]))
            aligned.append(x_aligned[0])
            names.append(dataset.idx_to_class[y])
    aligned = torch.stack(aligned).to(device)
    embeddings = resnet(aligned).detach().cpu()
    for i, eb in enumerate(embeddings): 
        face_db[names[i]] = eb
    with open(face_db_path, "wb") as f:
        pickle.dump(face_db, f)
    print('finished faceDatabase transform!', len(face_db.items()))
    return face_db


def load_face_db(known_images_path, face_db_path, device):
    if not os.path.exists(face_db_path):
        print('face_data_path not exist!,try to get faceDatabase transform!')
        face_db = update_face_db(known_images_path, device)
        return face_db
    with open(face_db_path, "rb") as f:
        face_db = pickle.load(f)
    print('finished load face_data!')
    return face_db

def mtcnn_recognition(img, face_db, unknown_embeddings, recog_thr, device) : 
    face_ids = []
    probs = []
    for i in range(len(unknown_embeddings)):
        result_dict = {}
        eb = unknown_embeddings[i]
        for (name) in face_db.keys() : 
            knownfeature = face_db[name]
            prob = (eb - knownfeature).norm().item()
            result_dict[name] = (prob)
        results = sorted(result_dict.items(), key=lambda d:d[1])
        result_name, result_probs = results[0][0], results[0][1]
        if result_probs < recog_thr: 
            face_ids.append(result_name)
        else : 
            face_ids.append('unknown')
        probs.append(result_probs)
    return face_ids, result_probs