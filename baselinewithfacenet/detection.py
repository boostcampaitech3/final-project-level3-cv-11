from util import CropRoiImg


def get_embeddings(resnet, img, bboxes, device, size = 160, save_path=None):
    faces = CropRoiImg(img, bboxes, size, save_path)
    # print(faces.shape)
    faces = faces.to(device)
    unknown_embeddings = resnet(faces).detach().cpu()
    return faces, unknown_embeddings


def recognizer(face_db, unknown_embeddings, recog_thr, known_ids) : 
    face_ids = []
    probs = []
    if known_ids is not None:
        iou_weights = known_ids.weights_list()
    for i in range(len(unknown_embeddings)):
        result_dict = {}
        eb = unknown_embeddings[i]
        if known_ids is not None:
            prev_name = iou_weights[i]
        for name in face_db.keys():
            knownfeature_list = face_db[name]
            prob_list = []
            for knownfeature in knownfeature_list:
                prob = (eb - knownfeature).norm().item()
                if known_ids is not None:
                    if prev_name == name:
                        prob -= known_ids.weight() # 가중치 값
                prob_list.append(prob)
                if prob < recog_thr:
                    # 기준 넘으면 바로 break해서 같은 인물 계속 안 체크하도록
                    break
            result_dict[name] = min(prob_list)
        results = sorted(result_dict.items(), key=lambda d:d[1])
        result_name, result_probs = results[0][0], results[0][1]
        if result_probs < recog_thr:
            face_ids.append(result_name)
        else: 
            face_ids.append('unknown')
        probs.append(result_probs)
    return face_ids, probs
