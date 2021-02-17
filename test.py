import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
import numpy as np

import models

def draw_text(ax, x, y, text, color):
    bbox_props = dict(boxstyle="round", fc='w', ec="0.5", alpha=0.4)
    ax.text(x, y, text,horizontalalignment='left',verticalalignment='bottom',fontsize=12, color=color, bbox=bbox_props)

parser = argparse.ArgumentParser()

parser.add_argument(
        '--image_file',
        type=str,
        help='Path to the image to test',
        required=True
        )

parser.add_argument(
        '--use_gpu',
        action='store_true',
        help='Whether to use GPU'
        )
parser.add_argument(
        '--model_file',
        type=str,
        help='Which pt model to load',
        required=True
)
parser.add_argument(
        '--target_mode',
        choices=['all_bbox', 'largest_bbox'],
        help='Which filter to apply to the targets',
        action='store',
        required=True
)
parser.add_argument(
    '--model',
    choices=['resnet18', 'resnet34','resnet50','resnet152','densenet121','squeezenet1_1'],
    help='Which pretrained model to use to compute the features',
    action='store',
    required=True
)

args = parser.parse_args()

device = torch.device('cpu')
if args.use_gpu:
    device = torch.device('cuda')

classes = ['Fish','Flower','Gravel','Sugar']

# Prepare the pipeline for preprocessing the image
imagenet_preprocessing = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

image_transform = transforms.Compose([transforms.Resize((700, 1050)),transforms.ToTensor(), imagenet_preprocessing])

# Prepare the model for processing the input tensor
print("Loading {}".format(args.model))
model_body = models.FeatureExtractor(model_name = args.model)
print("Loading the classification head")
model_head = torch.load(args.model_file)
model = nn.Sequential(model_body, model_head)

model = model.to(device=device)

# Put the model in test mode,This is super important given we may have batchnorm layers !
model.eval()

# Let us process the image
image = Image.open(args.image_file).convert('RGB')
input_tensor = image_transform(image).to(device=device).unsqueeze(0)

def plot_bbox(ax, bbox, text):
    cx, cy, width, height = bbox
    upl_x, upl_y = cx - width/2.0, cy - height/2.0
    p = patches.Rectangle( (upl_x, upl_y), width, height, fill=False, clip_on=False, edgecolor='yellow', linewidth=4 )
    ax.add_patch(p)
    draw_text(ax, upl_x, upl_y, text, 'yellow')

if args.target_mode == "largest_bbox":
    print("Forward propagating through the model")
    with torch.no_grad():
        predicted_bbox, logits = model(input_tensor)
    cx = predicted_bbox[0][0].item()
    cy = predicted_bbox[0][1].item()
    width = predicted_bbox[0][2].item()
    height = predicted_bbox[0][3].item()
    predicted_probas = F.softmax(logits, dim=1).squeeze()
    predicted_class = predicted_probas.argmax().item()
    predicted_proba = predicted_probas[predicted_class].item()

    print("Bounding box : (cx, cy, width, height) = ({},{},{},{})".format(cx, cy, width, height))
    print("Predicted probabilities {}".format(predicted_probas))

    plt.figure()
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Show the image
    plt.imshow(image, extent = [0, 1, 1, 0])

    # Overlay the bounding box
    plot_bbox(ax, (cx, cy, width, height), classes[predicted_class])

    plt.savefig("one_object_prediction.png", bbox_inches='tight')
    print("Image saved to one_object_prediction.png")

    plt.show()
