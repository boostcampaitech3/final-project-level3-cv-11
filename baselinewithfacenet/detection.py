from facenet_pytorch import MTCNN, InceptionResnetV1
import torch 
from torch.utils.data import DataLoader 
from torchvision import datasets
import os
import warnings 
import pickle
import cv2
import numpy as np
from PIL import Image

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


def build_face_db(known_images_path, face_db_path, img_db_path, device, args):
    face_db = {}
    if args['DETECTOR'] == 'retinaface':
        face_db_path += "_BGR"
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
        if args['DETECTOR'] == 'retinaface':
            x_np = np.array(x)
            x = cv2.cvtColor(x_np, cv2.COLOR_RGB2BGR)
        x_aligned, probs = mtcnn(x, return_prob=True)
        if x_aligned is not None: 
            print('Face detected with probability : {:8f}'.format(probs[0]))
            aligned.append(x_aligned[0])
            names.append(dataset.idx_to_class[y])
    aligned = torch.stack(aligned).to(device)
    embeddings = resnet(aligned).detach().cpu()
    for i, eb in enumerate(embeddings): 
        if names[i] not in face_db:
            face_db[names[i]] = [eb]
        else:
            face_db[names[i]].append(eb)
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


def load_face_db(known_images_path, face_db_path, img_db_path, device, args, model_args):
    if not os.path.exists(face_db_path):
        # 파일 자체가 없으면, 첫 기동
        print('face_data_path not exist!,try to get face Database transform!')
        face_db = build_face_db(known_images_path, face_db_path, img_db_path, device, args)
        return face_db
    else:
        new_img_list = check_face_db(known_images_path, img_db_path)
        if new_img_list:
            # 일부 파일 추가 필요
            print(len(new_img_list), 'files need get embedding! please wait...')
            mtcnn = model_args['Mtcnn']
            resnet = model_args['Recognition']
            face_db = get_embedding(face_db_path, new_img_list, mtcnn, resnet, device, args)
            with open(face_db_path, "wb") as f:
                pickle.dump(face_db, f)
        else:
            # 모든 파일 가지고 있음.
            with open(face_db_path, "rb") as f:
                face_db = pickle.load(f)
        
    print('finished load face_data!')
    return face_db


# def load_face_db(known_images_path, face_db_path, img_db_path, device, args):
#     if not os.path.exists(face_db_path):
#         print('face_data_path not exist!,try to get faceDatabase transform!')
#         face_db = build_face_db(known_images_path, face_db_path, img_db_path, device, args)
#         return face_db
#     with open(face_db_path, "rb") as f:
#         face_db = pickle.load(f)
#     print('finished load face_data!')
#     return face_db


def mtcnn_recognition(face_db, unknown_embeddings, recog_thr) : 
    face_ids = []
    probs = []
    for i in range(len(unknown_embeddings)):
        result_dict = {}
        eb = unknown_embeddings[i]
        for name in face_db.keys():
            knownfeature_list = face_db[name]
            prob_list = []
            for knownfeature in knownfeature_list:
                prob = (eb - knownfeature).norm().item()
                prob_list.append(prob)
                if prob < recog_thr:
                    # 기준 넘으면 바로 break해서 같은 인물 계속 안 체크하도록
                    break
            result_dict[name] = min(prob_list)
        results = sorted(result_dict.items(), key=lambda d:d[1])
        result_name, result_probs = results[0][0], results[0][1]
        if result_probs < recog_thr: 
            face_ids.append(result_name)
        else : 
            face_ids.append('unknown')
        probs.append(result_probs)
    return face_ids, result_probs