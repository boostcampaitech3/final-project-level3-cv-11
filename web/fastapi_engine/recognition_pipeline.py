from web.fastapi_engine.util import CropRoiImg


def mtcnn_get_embeddings(mtcnn, resnet, img, bboxes, device, save_path=None):
    faces = mtcnn.extract(img, bboxes, save_path)
    # print(faces.shape)
    faces = faces.to(device)
    unknown_embeddings = resnet(faces).detach().cpu()
    return faces, unknown_embeddings

def get_embeddings(resnet, img, bboxes, device, size=160, save_path=None):
    faces = CropRoiImg(img, bboxes, size, save_path)
    # print(faces.shape)
    faces = faces.to(device)
    unknown_embeddings = resnet(faces).detach().cpu()
    return faces, unknown_embeddings


def recognizer(face_db, unknown_embeddings, recog_thr) : 
    if len(face_db.keys()) == 0:
        face_ids = ['unknown'] * len(unknown_embeddings)
        probs = [0] * len(unknown_embeddings)
        return face_ids, probs
    
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
            result_dict[name] = min(prob_list)
        results = sorted(result_dict.items(), key=lambda d:d[1])
        result_name, result_probs = results[0][0], results[0][1]
        if result_probs < recog_thr: 
            face_ids.append(result_name)
        else : 
            face_ids.append('unknown')
        probs.append(result_probs)
    return face_ids, probs


# Recognition
def deepsort_recognizer(face_db, unknown_embeddings, recog_thr, id_name, identities):
    if len(face_db.keys()) == 0:
        id_name = ['unknown'] * len(unknown_embeddings)
        probs = [0] * len(unknown_embeddings)
        return id_name, probs
    
    probs = []
    for id_, emb in zip(identities, unknown_embeddings):
        result_dict = {}
        for name in face_db.keys():
            knownfeature_list = face_db[name]
            prob_list = []
            for knownfeature in knownfeature_list:
                prob = (emb - knownfeature).norm().item()
                prob_list.append(prob)
            result_dict[name] = min(prob_list)
        results = sorted(result_dict.items(), key=lambda d:d[1])
        result_name, result_probs = results[0][0], results[0][1]
        if result_probs < recog_thr:
            id_name[id_] = result_name
        else : 
            id_name[id_] =  'unknown'
    
    return id_name, probs
