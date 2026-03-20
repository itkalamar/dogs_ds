import torch 
import torchvision 
from torchvision import models 
from torchvision import transforms, datasets
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


model = models.squeezenet1_0(pretrained=True)

for param in model.parameters():
    param.required_grad = False 
    

num_classes  = 9
model.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1)),
    nn.ReLU(inplace=True),
    nn.AdaptiveAvgPool2d(output_size=(1,1))
)
model 

#data preparation

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225
])
    ])

#load dataset
import os
dataset_path = "c:\Dev\Dogs_dataset"

train_dataset = datasets.ImageFolder(
    root = os.path.join(dataset_path, "train")
    transform = transform
)

valid_dataset = datasets.ImageFolder(
    root = os.path.join(dataset_path, "valid"),
    transform = transform 
)


#dataLoder
train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_dl = DataLoader(valid_dataset, batch_size=32, shuffle=False)

print(f'Number_of_training samples: {len(train_dataset)}')
print(f'Number_of_validation samples: {len(valid_dataset)}')


#training setup 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001) 

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)