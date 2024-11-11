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
from wikiart import WikiArtDataset, WikiArtModel
import json
import argparse
import numpy as np
from imblearn.over_sampling import SMOTE  # for oversampling (adressing class imbalance)
from collections import Counter # for oversampling

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="configuration file", default="config.json")
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


def train(epochs=3, batch_size=32, modelfile=None, device="cpu"):
    loader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)

    ############################################################################################
    # ADDRESSING DATA IMBALANCE PROBLEM WITH SMOTE:
    X_train = []
    y_train = []
    for image, label in loader:
        X_train.append(image.cpu().numpy()) # image
        y_train.extend(label.numpy()) # label
        #print("done")
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.array(y_train)
    
    sampling_strategy = {} # sampling strategy will be dict due to large size of dataset
    label_counts = Counter(y_train)
    
    # we need to check the n of instances per class since oversampling will not work if the n of instances wanted is lower than the existing n of instances  
    for label in label_counts:
        if label_counts[label] > 200:
            sampling_strategy[label] = label_counts[label]  # use existing if n of instances surpasses 200
        else:
            sampling_strategy[label] = 200  # use 200 if not
            
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    
    # we need to flatten X_train since SMOTE only accepts dimensions of <=2
    X_train_flattened = X_train.reshape(X_train.shape[0], -1)
    print("doing upsampling....please wait...")
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_flattened, y_train)
    print("done with resampling")
    print("this is y_train after the SMOTE process", Counter(y_train_resampled))

    # back into tensors
    X_train_resampled = X_train_resampled.reshape(-1, 3, 416, 416) # changing the input back into the original dimensions
    X_train_resampled = torch.tensor(X_train_resampled)
    y_train_resampled = torch.tensor(y_train_resampled)
    
    # new dataloader needed for the training loop
    train_dataset = TensorDataset(X_train_resampled, y_train_resampled)
    loader_resampled = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)



    ###################################################################################################

    model = WikiArtModel().to(device)
    optimizer = Adam(model.parameters(), lr=0.001) #learning rate changed from 0.01
    criterion = nn.NLLLoss().to(device)
    
    for epoch in range(epochs):
        print("Starting epoch {}".format(epoch))
        accumulate_loss = 0
        for batch_id, batch in enumerate(tqdm.tqdm(loader_resampled)):
            X, y = batch
            y = y.to(device)
            X = X.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            accumulate_loss += loss
            optimizer.step()

        print("In epoch {}, loss = {}".format(epoch, accumulate_loss))

    if modelfile:
        torch.save(model.state_dict(), modelfile)

    return model

model = train(epochs, config["batch_size"], modelfile=config["modelfile"], device=device)