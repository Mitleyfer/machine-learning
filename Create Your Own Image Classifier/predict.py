import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torchvision import datasets, models, transforms
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import argparse
import json

parser = argparse.ArgumentParser(description="predict.py")
parser.add_argument('--checkpoint', action='store', default='checkpoint_resnet18.pth', help='Set path to trained network')
parser.add_argument('--top_k', default=5, help='Top k numbers')
parser.add_argument('--category_names', default='./cat_to_name.json', help='Use a mapping of categories to real name')
parser.add_argument('--file', dest='file', default='flowers/test/88/image_00540.jpg', help='Set path to image')
parser.add_argument('--GPU', action='store', default='GPU', help='Use GPU for predictions')

arg = parser.parse_args()
checkpoint=arg.checkpoint
file = arg.file
top_k=arg.top_k
dev = arg.GPU
cat_names=arg.category_names

if dev=='GPU':
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

with open(cat_names, 'r') as f:
    cat_to_name = json.load(f)
    
def loading_model(path):
    checkpoint = torch.load(path)
    if checkpoint['pretrained_model']=='vgg11':
        model = models.vgg11(pretrained=True)
        model.classifier = checkpoint['classifier']
    elif checkpoint['pretrained_model']=='AlexNet':
        model = models.AlexNet(pretrained=True)
        model.classifier = checkpoint['classifier']
    else:
        model = models.resnet18(pretrained=True)
        model.fc = checkpoint['classifier']
    learning_rate = checkpoint['learning_rate']
    model.epochs = checkpoint['epochs']
    model.optimizer = checkpoint['optimizer']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['model'])
    criterion = checkpoint['loss']
    
    return model

model_trained = loading_model(checkpoint)

def process_image_PyTorch(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    resizing = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    result = resizing(image)
    return np.array(result)

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to(device)
    model.eval()
    
    img_open = Image.open(image_path) 
    img_processed = process_image_PyTorch(img_open)
    img_torch = torch.from_numpy(img_processed)
    
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()
    
    if device == torch.device('cuda'):
        with torch.no_grad():
            output = model.forward(img_torch.cuda())
    else: 
        with torch.no_grad():
            output = model.forward(img_torch)
        
    probability = F.softmax(output.data,dim=1)
    
    probs = np.array(probability.topk(topk)[0][0])
    
    index_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [np.int(index_to_class[each]) for each in np.array(probability.topk(topk)[1][0])]
    
    return probs, top_classes

path = file
image = process_image_PyTorch(Image.open(path))    
probs, classes = predict(path, model_trained, int(top_k))
names = [cat_to_name[str(index)] for index in classes]

for i in range(len(names)):
    print('Image is {} with probability {}'.format(names[i],probs[i]))