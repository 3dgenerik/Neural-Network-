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
test_loader = torch.utils.data.DataLoader(testset, batch_size = 32)
valid_loader = torch.utils.data.DataLoader(validset, batch_size = 32)

input_layer = 224*224
hidden_layers = [4096, 2048, 1024, 512]
output_layer = 102 


class Network(nn.Module):
    def __init__(self, input_layer, hidden_layers, output_layer, drop_p = 0.5):
        super().__init__()
        
        self.hidden_layers = nn.ModuleList([nn.Linear(input_layer, hidden_layers[0])])
        layer_size = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_size])
        self.output = nn.Linear(hidden_layers[-1], output_layer)
        self.dropout = nn.Dropout(p = drop_p)
        
    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        x = self.output(x)
        return  F.log_softmax(x, dim = 1)


model = Network(input_layer, hidden_layers, output_layer)
  
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

def validation(model, input_loader, device, input_layer, criterion):
    test_loss = 0
    accuracy = 0
    model.to(device)
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        images.resize_(images.shape[0], input_layer)
        
        output = model.forward(images)
        loss = criterion(output, labels)
        
        test_loss += loss.item()
        
        ps = torch.exp(output)
        
        prediction = torch.max(ps, dim = 1)[1]
        labeling = (prediction == labels)
        
        accuracy += labeling.type(torch.FloatTensor).mean()
    return test_loss, accuracy


epochs = 3
print_every = 20
steps = 0

model.to(device)

for e in range(epochs):
    running_error = 0
    model.train()
    for images, labels in train_loader:
        steps += 1
        images, labels = images.to(device), labels.to(device)
        images.resize_(images.shape[0], input_layer)
        
        optimizer.zero_grad()
        
        output = model.forward(images)
        loss = criterion(output, labels)
        
        
        loss.backward()
        optimizer.step()
        
        running_error += loss.item()
        
        if steps%print_every == 0:
            model.eval()
            with torch.no_grad():
                test_loss, accuracy = validation(model, test_loader, device, input_layer, criterion)
            
            print("Epochs {}/{}".format(e+1, epochs),
                  "Loss {:.3f}".format(running_error/print_every),
                  "Validation loss {:.4f}".format(test_loss/len(test_loader)),
                  "Validation accuracy {:.4f}".format(accuracy/len(test_loader)))
            running_error = 0
            model.train()
            


def get_accuracy(model, images_loader, device, input_layer):
    correct = 0
    total = 0
    model.to(device)
    with torch.no_grad():
        for images, labels in images_loader:
            images, labels = images.to(device), labels.to(device)
            images.resize_(images.shape[0], input_layer)
            
            output = model.forward(images)
            total += labels.size()[0]
            
            predictions = torch.max(output, dim = 1)[1]
            labeling = (predictions == labels)
            
            correct += labeling.sum().item()
    
    print("Accuracy of the test data:: {}%".format(100 * correct/total))   
        
get_accuracy(model, test_loader, device, input_layer)

#######################################################################
#---------------------------------OUTPUT------------------------------#
#######################################################################
"""
Epochs 1/3 Loss 7.436 Validation loss 4.6647 Validation accuracy 0.0024
Epochs 1/3 Loss 4.961 Validation loss 4.6119 Validation accuracy 0.0240
Epochs 1/3 Loss 4.753 Validation loss 4.6207 Validation accuracy 0.0276
Epochs 1/3 Loss 4.681 Validation loss 4.6156 Validation accuracy 0.0288
Epochs 1/3 Loss 4.666 Validation loss 4.6133 Validation accuracy 0.0325
Epochs 2/3 Loss 3.935 Validation loss 4.6081 Validation accuracy 0.0240
Epochs 2/3 Loss 4.645 Validation loss 4.6091 Validation accuracy 0.0276
Epochs 2/3 Loss 4.643 Validation loss 4.6073 Validation accuracy 0.0216
Epochs 2/3 Loss 4.621 Validation loss 4.6010 Validation accuracy 0.0264
Epochs 2/3 Loss 4.623 Validation loss 4.6074 Validation accuracy 0.0288
Epochs 3/3 Loss 3.239 Validation loss 4.5966 Validation accuracy 0.0192
Epochs 3/3 Loss 4.619 Validation loss 4.5977 Validation accuracy 0.0288
Epochs 3/3 Loss 4.608 Validation loss 4.5956 Validation accuracy 0.0276
Epochs 3/3 Loss 4.615 Validation loss 4.5930 Validation accuracy 0.0288
Epochs 3/3 Loss 4.618 Validation loss 4.6231 Validation accuracy 0.0385


Accuracy of the test data: 2.076 %
"""

