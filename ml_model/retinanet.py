import io
import numpy as np
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from PIL import ImageDraw, ImageFont
from torchvision.models.detection import retinanet_resnet50_fpn


COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def plausible(pred, threshold=0.5):

    boxes = pred["boxes"].cpu().detach().numpy()
    labels = pred["labels"].cpu().detach().numpy()

    if "scores" in pred.keys():
        scores = pred["scores"].cpu().detach().numpy()
        boxes = boxes[scores > threshold]
        labels = labels[scores > threshold]

    labels = np.vectorize(lambda x: COCO_INSTANCE_CATEGORY_NAMES[x])(labels)
    return {"boxes": boxes.tolist(), "labels": labels.tolist()}


def prediction(image_data):
    img = Image.open(io.BytesIO(image_data))
    transform = transforms.ToTensor()
    x = transform(img)

    model = retinanet_resnet50_fpn(pretrained=True)
    model.eval()
    y = model(x.unsqueeze(0))[0]
    return plausible(y)

