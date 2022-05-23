import torch
import argparse

class Args(object):
    parser = argparse.ArgumentParser(description='Arguments for Unknown mosaic')
    parser.add_argument('--image_dir', help='Directory to image')
    parser.add_argument('--bbox_thrs', type=int, default=30, help='Threshold of bounding box')
    parser.add_argument('--recog_thrs', type=int, default=30, help='Threshold of recognition')

    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for data split')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--save_dir', default='./saved', help='Directory to save model')
    parser.add_argument('--pretrained_path', default=None, help='Pre-trained model path') # Train-> Fine tuning, Test-> Inference


    parse = parser.parse_args()
    params = {
        "IMAGE_DIR": parse.model, 
        "BBOX_THRESHOLD": parse.resize, 
        "RECOG_THRESHOLD": parse.learning_rate,
        
        "NUM_WORKERS": parse.num_workers,
        "RANDOM_SEED": parse.random_seed,
        "DEVICE": parse.device,
        "SAVE_DIR": parse.save_dir, 
        "PRETRAINED_PATH": parse.pretrained_path,
    }