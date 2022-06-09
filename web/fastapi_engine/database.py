import os
import pickle
from pathlib import Path

def load_face_db(db_path):
    parent_path = str(Path(db_path).parent.absolute())
    if not os.path.exists(parent_path):
        os.mkdir(parent_path)
    if not os.path.exists(db_path):
        os.mkdir(db_path)

    face_db_path = os.path.join(db_path, 'face_db')

    if not os.path.exists(face_db_path):
        # 파일 자체가 없으면, 첫 기동
        print('face_data_path not exist!,try to get face Database transform!')
        with open(face_db_path, "wb") as f:
            pickle.dump(dict(), f)
        return dict()

    else:
        with open(face_db_path, "rb") as f:
            face_db = pickle.load(f)
        
        print('finished load face_data!')
        return face_db
