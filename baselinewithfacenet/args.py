import torch
import argparse

class Args(object):
    parser = argparse.ArgumentParser(description='Arguments for Unknown mosaic')
    parser.add_argument('--process_target', type=str, default='Video', help='Image, Video, Cam')
    parser.add_argument('--image_dir', default="../data/dest_images/findobama/twopeople.jpeg", help='Directory to image')
    parser.add_argument('--bbox_thrs', type=int, default=30, help='Threshold of bounding box')
    parser.add_argument('--recog_thrs', type=float, default=0.8, help='Threshold of recognition')
    parser.add_argument('--input_mode', type=str, default='tv', help='cv2, PIL, tv: torchvision')
    parser.add_argument('--detector', type=str, default='mtcnn', help='Detection model')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for data split')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--save_dir', default='./saved', help='Directory to save model')
    parser.add_argument('--pretrained_path', default=None, help='Pre-trained model path') # Train-> Fine tuning, Test-> Inference
    parser.add_argument('--debug_mode', type=bool, default=False, help='print on off')


    parse = parser.parse_args()
    params = {
        "PROCESS_TARGET": parse.process_target,
        "IMAGE_DIR": parse.image_dir, 
        "BBOX_THRESHOLD": parse.bbox_thrs, 
        "RECOG_THRESHOLD": parse.recog_thrs,
        "INPUT_MODE": parse.input_mode,
        "DETECTOR": parse.detector,
        "NUM_WORKERS": parse.num_workers,
        "RANDOM_SEED": parse.random_seed,
        "DEVICE": parse.device,
        "SAVE_DIR": parse.save_dir, 
        "PRETRAINED_PATH": parse.pretrained_path,
        "DEBUG_MODE": parse.debug_mode,
    }