import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import json
from zmq.backend.cython import _device

main_folder = "flowers"
train_dir = main_folder + "/train"
test_dir = main_folder + "/test"
valid_dir = main_folder + "/valid"

device = "cuda"


train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomRotation(30),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.RandomResizedCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

valid_transforms = test_transforms

train_set = datasets.ImageFolder(train_dir, transform=train_transform)
test_set = datasets.ImageFolder(test_dir, transform=test_transforms)
valid_set = datasets.ImageFolder(valid_dir, transform=valid_transforms)

train_loader = torch.utils.data.DataLoader(train_set, batch_size = 64, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = 32)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size = 32)


model = models.vgg16(pretrained = True)

for param in model.parameters():
    param.request_grad = False

classifier = nn.Sequential(OrderedDict([("fc_1", nn.Linear(25088, 1024)),
                                        ("relu_1", nn.ReLU()),
                                        ("dropout_1", nn.Dropout(p = 0.5)),
                                        ("fc_2", nn.Linear(1024, 102)),
                                        ("output", nn.LogSoftmax(dim = 1))]))


model.classifier = classifier

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = 0.001)

def validation(device, model, test_loader, criterion):
    test_loss = 0
    accuracy = 0
    model.to(device)
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
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
loss_graph = []
accuracy_graph = []

model.to(device)

for e in range(epochs):
    running_error = 0
    
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        steps += 1
        
        optimizer.zero_grad()
        
        output = model.forward(images)
        loss = criterion(output, labels)
        
        loss.backward()
        optimizer.step()
        
        running_error += loss.item()
        
        if steps%print_every == 0:
            model.eval()
            with torch.no_grad():
                test_loss, accuracy = validation(device, model, test_loader, criterion)
            
            test_loss_percentage = test_loss/len(test_loader)
            accuracy_percentage = 100*accuracy/len(test_loader)
            print("Epochs: {}/{} ".format(e+1, epochs),
                  "Loss: {:.3f} ".format(running_error/print_every),
                  "Valid_loss: {:.3f} ".format(test_loss_percentage),
                  "Valid_accuracy: {:.3f} ".format(accuracy_percentage))
            accuracy_graph.append(accuracy_percentage)
            loss_graph.append(test_loss_percentage)
            running_error = 0
            model.train()

plt.plot(accuracy_graph)
plt.plot(loss_graph)
plt.show()






def test_valid_accuracy(device, model, image_loader):
    correct = 0
    total = 0
    model.to(device)
    for images, labels in image_loader:
        images, labels = images.to(device), labels.to(device)
        
        output = model.forward(images)
        
        total += labels.size()[0]
        prediction = torch.max(output, dim = 1)[1]
        labeling = (prediction==labels)
        correct += labeling.sum().item()
        print("Valid/test set temp accuracy: {:.3f}".format(100*correct/total))
    print("Valid/test set main accuracy: {:.3f}".format(100*correct/total))



########################################################################################
##############################S AVE AND LOAD NETWORK ###################################

print(model.classifier)

"""

  )
  (classifier): Sequential(
    (fc_1): Linear(in_features=25088, out_features=1024, bias=True)
    (relu_1): ReLU()
    (dropout_1): Dropout(p=0.5)
    (fc_2): Linear(in_features=1024, out_features=102, bias=True)
    (output): LogSoftmax()
  )
)
odict_keys(['features.0.weight', 'features.0.bias', 'features.2.weight', 'features.2.bias',
            'features.5.weight', 'features.5.bias', 'features.7.weight', 'features.7.bias',
            'features.10.weight', 'features.10.bias', 'features.12.weight', 'features.12.bias',
            'features.14.weight', 'features.14.bias', 'features.17.weight', 'features.17.bias',
            'features.19.weight', 'features.19.bias', 'features.21.weight', 'features.21.bias',
            'features.24.weight', 'features.24.bias', 'features.26.weight', 'features.26.bias',
            'features.28.weight', 'features.28.bias', 'classifier.fc_1.weight', 'classifier.fc_1.bias',
            'classifier.fc_2.weight', 'classifier.fc_2.bias'])
"""


filename = "flower_checkpoint_2.pth"

checkpoint = {"fc_1": (25088, 1024),
              "fc_2":(1024,102),
              "state_dict":model.state_dict(),
              "optimizer": optimizer.state_dict()}

torch.save(checkpoint, filename)

def load_checkpoint(filename, optimizer, model):
    checkpoint = torch.load(filename)
    model = model.classifier(checkpoint["fc_1"], checkpoint["fc_2"])
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return model

model = load_checkpoint(filename, optimizer, model)
print(model)

#output  
"""
Traceback (most recent call last):
  File "C:\Users\CGID\eclipse-workspace\Udacity_course\IMAGE_CLASSIFIER_PROJECT_FINAL\aipnd-project-master\train.py", line 142, in <module>
    model = load_checkpoint(filename, optimizer, model)
  File "C:\Users\CGID\eclipse-workspace\Udacity_course\IMAGE_CLASSIFIER_PROJECT_FINAL\aipnd-project-master\train.py", line 137, in load_checkpoint
    model = model.classifier(checkpoint["fc_1"], checkpoint["fc_2"])
  File "C:\Users\CGID\Anaconda3\lib\site-packages\torch\nn\modules\module.py", line 491, in __call__
    result = self.forward(*input, **kwargs)
TypeError: forward() takes 2 positional arguments but 3 were given
"""    
  