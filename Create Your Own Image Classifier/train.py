import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torchvision import datasets, models, transforms
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="train.py")
parser.add_argument('--save_dir', dest="save_dir", action="store", default="checkpoint.pth", help='Save classifier')
parser.add_argument('--arch', dest='arch', default='resnet18', choices=['vgg11', 'resnet18', 'AlexNet'],help='Available pretrained networks are: vgg11, resnet18, AlexNet')
parser.add_argument('--learning_rate', dest='learning_rate', default='0.002', help='Set learning rate')
parser.add_argument('--hidden_units', dest='hidden_units', default='256', help='Set number of hidden units in classifier')
parser.add_argument('--epochs', dest='epochs', default='3', help='Set number of epochs')
parser.add_argument('--GPU', action='store', default='GPU', help='Use GPU for training')
    
arg = parser.parse_args()
architecture = arg.arch
dev = arg.GPU
hidden_layers = arg.hidden_units
epochs = arg.epochs
learning_rate = arg.learning_rate
save_dir = arg.save_dir


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

data_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
random_transforms = [transforms.RandomAffine(30), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
train_transforms = transforms.Compose([transforms.RandomRotation(45),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomChoice(random_transforms),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

train_dataset = datasets.ImageFolder(train_dir, transform = train_transforms)
valid_dataset = datasets.ImageFolder(valid_dir, transform = data_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform = data_transforms)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

if dev == 'GPU':
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
if architecture == 'AlexNet':
    model = models.AlexNet(pretrained=True)
    classifier = nn.Sequential(OrderedDict([('inp', nn.Linear(256,int(hidden_layers))),
                                       ('dropout', nn.Dropout(p=0.4)),
                                       ('relu', nn.ReLU()),
                                       ('out', nn.Linear(int(hidden_layers),102)),
                                       ('softmax', nn.LogSoftmax(dim=1))]))
    for i in model.parameters():
        i.requires_grad = False
    
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = float(learning_rate))
elif architecture == 'vgg11':
    model = models.vgg11(pretrained=True)
    classifier = nn.Sequential(OrderedDict([('inp', nn.Linear(25088,int(hidden_layers))),
                                       ('dropout', nn.Dropout(p=0.4)),
                                       ('relu', nn.ReLU()),
                                       ('out', nn.Linear(int(hidden_layers),102)),
                                       ('softmax', nn.LogSoftmax(dim=1))]))
    for i in model.parameters():
        i.requires_grad = False
    
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = float(learning_rate))
else:
    model = models.resnet18(pretrained=True)
    classifier = nn.Sequential(OrderedDict([('inp', nn.Linear(512,int(hidden_layers))),
                                       ('dropout', nn.Dropout(p=0.4)),
                                       ('relu', nn.ReLU()),
                                       ('out', nn.Linear(int(hidden_layers),102)),
                                       ('softmax', nn.LogSoftmax(dim=1))]))
    for i in model.parameters():
        i.requires_grad = False
    
    model.fc = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr = float(learning_rate))
    
model.to(device)
epochs = int(epochs)
steps = 0
running_loss = 0
printing = 10
i = 0
control_dict = {}

for epoch in range(epochs):
    for inputs, labels in trainloader:
        optimizer.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        i+=1
        if steps%printing == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            
            for inputs1, labels1 in validloader:
                optimizer.zero_grad()
                inputs1, labels1 = inputs1.to(device), labels1.to(device)
                model.to(device)
                with torch.no_grad():
                    logps = model.forward(inputs1)
                    batch_loss = criterion(logps, labels1)
                    valid_loss += batch_loss.item()
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equality = top_class == labels1.view(*top_class.shape)
                    accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
                    
            control_dict[i] = [running_loss/printing, valid_loss/len(validloader), accuracy/len(validloader)]
        
            print(f"{i}) Epoch {epoch+1}/{epochs}.. "
                f"Train loss: {running_loss/printing:.3f}.. "
                f"Test loss: {valid_loss/len(validloader):.3f}.. "
                f"Test accuracy: {accuracy/len(validloader):.3f}")
            running_loss = 0
            model.train()
            
loss = 0
accuracy = 0
model.to(device)
model.eval()
with torch.no_grad():
    for inputs2, labels2 in testloader:
        inputs2, labels2 = inputs2.to(device), labels2.to(device)
        test_logps = model.forward(inputs2)
        test_ps = torch.exp(test_logps)
        top_p, top_class = test_ps.topk(1, dim=1)
        equals = top_class == labels2.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

print('Test set accuracy: %d %%' % (100 * accuracy / len(testloader)))

model.class_to_idx = train_dataset.class_to_idx
    
checkpoint = OrderedDict([('epochs', epochs),
                          ('model', model.state_dict()),
                          ('classifier', classifier),
                          ('optimizer', optimizer.state_dict()),
                          ('loss', criterion),
                          ('learning_rate', learning_rate),
                          ('class_to_idx', model.class_to_idx),
                          ('pretrained_model', architecture)])
torch.save(checkpoint, save_dir)

print("Your model has been trained and saved")