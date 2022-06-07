from pathlib import Path
import sys
path = Path(__file__)
sys.path.append(str(path.parent.absolute()))

from face_detection.yolov5_face.load_yolov5 import load_yolov5