import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import json


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

device = "cuda"


transform_train = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomRotation(30),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

transform_test = transforms.Compose([transforms.Resize(256),
                                     transforms.RandomResizedCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

transform_valid = transform_test

trainset = datasets.ImageFolder(train_dir, transform = transform_train)
testset = datasets.ImageFolder(test_dir, transform = transform_test)
validset = datasets.ImageFolder(valid_dir, transform = transform_valid)

train_loader = torch.utils.data.DataLoader(trainset, batch_size = 64, shuffle = True)
test_loader = torch.utils.data.DataLoader(testset, batch_size = 64, shuffle = True)
valid_loader = torch.utils.data.DataLoader(validset, batch_size = 64, shuffle = True)


model = models.vgg11(pretrained = True)

for param in model.parameters():
    param.required_grad = False
    
    
classifier = nn.Sequential(OrderedDict([("fc_1", nn.Linear(25088, 2048)),
                                        ("relu_1", nn.ReLU()),
                                        ("dropout_1", nn.Dropout(p = 0.5)),
                                        ("fc_2", nn.Linear(2048, 1024)),
                                        ("relu_2", nn.ReLU()),
                                        ("dropout_2",nn.Dropout(p = 0.5)),
                                        ("fc_3", nn.Linear(1024, 512)),
                                        ("relu_3", nn.ReLU()),
                                        ("dropout_3", nn.Dropout(p = 0.5)),
                                        ("fc_4", nn.Linear(512, 102)),
                                        ("relu_4", nn.ReLU()),
                                        ("dropout_4", nn.Dropout(p = 0.5)),
                                        ("output", nn.LogSoftmax(dim = 1))]))

model.classifier = classifier

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = 0.001)

epochs = 2
print_every = 20
steps = 0


model.to(device)
for e in range(epochs):
    running_error = 0
    
    for images, labels in train_loader:
        steps += 1
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        output = model.forward(images)
        loss = criterion(output, labels)
        
        loss.backward()
        optimizer.step()
        
        running_error += loss.item()
        
        if steps%print_every == 0:
            print("Epochs {}/{}".format(e+1, epochs),
                  "Loss {:.3f}".format(running_error/print_every))
            running_error = 0
