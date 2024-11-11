import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torch.optim import Adam
import tqdm


class WikiArtImage:
    def __init__(self, imgdir, label, filename):
        self.imgdir = imgdir
        self.label = label
        self.filename = filename
        self.image = None
        self.loaded = False

    def get(self):
        if not self.loaded:
            self.image = read_image(os.path.join(self.imgdir, self.label, self.filename)).float()
            self.loaded = True

        return self.image

class WikiArtDataset(Dataset):
    def __init__(self, imgdir, device="cpu"):
        walking = os.walk(imgdir)
        filedict = {}
        indices = []
        classes = set()
        n_of_artfiles = []
        print("Gathering files for {}".format(imgdir))
        for item in walking:
            sys.stdout.write('.')
            arttype = os.path.basename(item[0])
            artfiles = item[2]
            #print("this is the art type", arttype)
            n_of_artfiles.append(len(artfiles))
            for art in artfiles:
                filedict[art] = WikiArtImage(imgdir, arttype, art)
                indices.append(art)
                classes.add(arttype)  
        print("...finished")
        self.filedict = filedict
        self.imgdir = imgdir
        self.indices = indices
        self.classes = list(sorted(classes))
        print(self.classes)
        self.device = device
        print(sum(n_of_artfiles)/len(n_of_artfiles))
        
    def __len__(self):
        return len(self.filedict)

    def __getitem__(self, idx):
        #print(len(self))
        imgname = self.indices[idx]
        imgobj = self.filedict[imgname]
        ilabel = self.classes.index(imgobj.label)
        image = imgobj.get().to(self.device)

        return image, ilabel

class WikiArtModel(nn.Module):
    def __init__(self, num_classes=27):
        super().__init__()
        # convolutional layers in order
        self.conv2d1 = nn.Conv2d(3, 16, (2,2), padding=2) # n of out channels changed from 1 to 16, but kept kernel size at (2,2)
        self.conv2d2 = nn.Conv2d(16, 32, (2,2), padding=2) # added this second convolutional layer
        self.conv2d3 = nn.Conv2d(32, 64, (2,2), padding=2) #added this third convolutional layer

        # batch normalization layers in order
        self.batchnorm2d1 = nn.BatchNorm2d(16) #changed from (105x105) and also changed to BatchNorm2d
        self.batchnorm2d2 = nn.BatchNorm2d(32) #added
        self.batchnorm2d3 = nn.BatchNorm2d(64) #added
        
        # max pooling
        self.maxpool2d = nn.MaxPool2d((4,4), padding=2) 

        # flattening layer
        self.flatten = nn.Flatten()

        # linear layers
        self.linear1 = nn.Linear(4096, 300) #changed from ((105x105), 300) 
        self.linear2 = nn.Linear(300, num_classes)

        #dropout, relu and softmax
        self.dropout = nn.Dropout(0.1) #changed droupout rate from 0.01 to 0.1
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, image):
        output = self.conv2d1(image)
        #print("output size after first convolutional layer {}".format(output.size()))
        output = self.batchnorm2d1(output)
        #print("output size after first batch normalization {}".format(output.size()))
        output = self.relu(output)
        #print("output size after first relu {}".format(output.size()))
        output = self.maxpool2d(output)
        #print("output size after first maxpool {}".format(output.size()))
        
        output = self.conv2d2(output)
        #print("output size after second convolutional layer {}".format(output.size()))
        output = self.batchnorm2d2(output)
        #print("output size after second batch normalization {}".format(output.size()))
        output = self.relu(output)
        #print("output size after second relu {}".format(output.size()))
        output = self.maxpool2d(output)
        #print("output size after second maxpool {}".format(output.size()))

        output = self.conv2d3(output)
        #print("output size after third convolutional layer {}".format(output.size()))
        output = self.batchnorm2d3(output)
        #print("output size after third batch normalization {}".format(output.size()))
        output = self.relu(output)
        #print("output size after third relu {}".format(output.size()))
        output = self.maxpool2d(output)
        #print("output size after third maxpool {}".format(output.size()))
        
        output = self.flatten(output)
        #print("size of the output after the flattening layer {}".format(output.size()))
            
        output = self.linear1(output)
        output = self.dropout(output)
        output = self.relu(output)
        output = self.linear2(output)
        return self.softmax(output)


