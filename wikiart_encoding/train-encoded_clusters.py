import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.io import read_image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torch.optim import Adam
import tqdm
from wikiart_encoded_clusters import WikiArtDataset, WikiArtModel
import json
import argparse
import numpy as np
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE  # for oversampling (adressing class imbalance)
from collections import Counter # for oversampling

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="configuration file", default="config-encoded_clusters.json")
parser.add_argument('epochs', type=int, help="The number of epochs for the training loop.")

args = parser.parse_args()

config = json.load(open(args.config))
epochs = args.epochs

trainingdir = config["trainingdir"]
testingdir = config["testingdir"]
device = config["device"]

print("Running...")


traindataset = WikiArtDataset(trainingdir, device)

#testingdataset = WikiArtDataset(testingdir, device)

#print(traindataset.imgdir)
the_image, the_label = traindataset[5]
#print(the_image, the_image.size())

def train(epochs=epochs, batch_size=32, modelfile=None, device="cpu"):
    """
    Trains the encoder model that produces compressed representations of image data.
    """
    loader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)  
    model = WikiArtModel().to(device)
    optimizer = Adam(model.parameters(), lr=0.001) 
    criterion = nn.MSELoss().to(device)
    
    for epoch in range(epochs):
        print("Starting epoch {}".format(epoch))
        accumulate_loss = 0
        for batch_id, batch in enumerate(tqdm.tqdm(loader)): 
            X, y = batch
            #X = X.to(device)
            #y = y.to(device)
            optimizer.zero_grad()
            output = model(X)  #passing the input through the encoder AND the decoder
            loss = criterion(output, X)
            loss.backward()
            accumulate_loss += loss.item()
            optimizer.step()

        print("In epoch {}, loss = {}".format(epoch, accumulate_loss))

    if modelfile:
        torch.save(model.state_dict(), modelfile)

    return model

model = train(epochs, config["batch_size"], modelfile=config["modelfile"], device=device)
