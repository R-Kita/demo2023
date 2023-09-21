import numpy as np
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from PIL import ImageDraw, ImageFont
import matplotlib.pyplot as plt
from torchvision.models.detection import retinanet_resnet50_fpn

def prediction(image_data):
    return {"retinanet": "test"}

