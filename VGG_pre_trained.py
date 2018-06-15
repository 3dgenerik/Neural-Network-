import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import json

#image folder path
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

device = "cuda"

#transforms
train_transform = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

validation_transform = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_transform = validation_transform

#datasets
train_dataset = datasets.ImageFolder(train_dir, transform = train_transform)
validation_dataset = datasets.ImageFolder(valid_dir, transform = validation_transform)
test_dataset = datasets.ImageFolder(test_dir, transform = test_transform)

#data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle = True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size = 64)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 64)


#get labels with .json
with open("cat_to_name.json") as f:
    cat_to_name = json.load(f)

#model.classifier = Network(input_layer, hidden_layers, output_layer = 0.5)
#x = Network(input_layer, hidden_layers, output_layer, 0.5)

model = models.vgg11(pretrained = True)

classifier = nn.Sequential(OrderedDict([("fc1", nn.Linear(25088, 2048)),
                                        ("relu", nn.ReLU()),
                                        ("dropout1", nn.Dropout(0.5)),
                                        ("fc2", nn.Linear(2048, 1024)),
                                        ("relu2", nn.ReLU()),
                                        ("dropout2", nn.Dropout(0.5)),
                                        ("fc3", nn.Linear(1024, 512)),
                                        ("relu3", nn.ReLU()),
                                        ("dropout3", nn.Dropout(0.5)),
                                        ("output", nn.Linear(512, 102))]))

model.classifier = classifier



#cross entropy and SGD
#criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.classifier.parameters(), lr = 0.03)

#NLLLoss and Adam
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = 0.03)

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
            print("Eposhs: {}/{}".format(e+1, epochs),
                  "Loss: {:.3f}".format(running_error/print_every))
            running_error = 0 


