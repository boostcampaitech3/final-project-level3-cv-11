# Facenet - pytorch 

## 1. Installation 
Reference : [FaceNet-Pytorch](https://github.com/timesler/facenet-pytorch)
```shell
# With pip:
pip install facenet-pytorch

# or clone this repo, removing the '-' to allow python imports:
git clone https://github.com/timesler/facenet-pytorch.git facenet_pytorch

# or use a docker container (see https://github.com/timesler/docker-jupyter-dl-gpu):
docker run -it --rm timesler/jupyter-dl-gpu pip install facenet-pytorch && ipython
```

Use it like... 
```python
from facenet_pytorch import MTCNN, InceptionResnetV1

# If required, create a face detection pipeline using MTCNN:
mtcnn = MTCNN(image_size=<image_size>, margin=<margin>)

# Create an inception resnet (in eval mode):
resnet = InceptionResnetV1(pretrained='vggface2').eval()
```

If error like this appears.. 
`RuntimeError: Couldn't load custom C++ ops. This can happen if your Pytorch and torchvision versions are incompatible ... `
```shell
# if python 3.8
pip uninstall torch torchvision 
pip install torch==1.7.1 torchvision==0.8.2
```


