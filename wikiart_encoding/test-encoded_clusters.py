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
from wikiart_encoded_clusters import WikiArtDataset, WikiArtModel
import torcheval.metrics as metrics
import json
import argparse
from sklearn.cluster import KMeans # for clustering
from sklearn.decomposition import PCA # for clustering
import matplotlib.pyplot as plt # for plotting
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="configuration file", default="config-encoded_clusters.json")

args = parser.parse_args()

config = json.load(open(args.config))

testingdir = config["testingdir"]
device = config["device"]


print("Running...")

#traindataset = WikiArtDataset(trainingdir, device)
testingdataset = WikiArtDataset(testingdir, device)

def test(modelfile=None, device="cpu"):
    """
    The inputs area passed through the encoder of the model.
    The compressed representations of the inputs are clustered, 
    their dimensionality is reduced with PCA, and then they are plotted on a graph.
    """
    loader = DataLoader(testingdataset, batch_size=1)

    model = WikiArtModel()
    model.load_state_dict(torch.load(modelfile, weights_only=True))
    model = model.to(device)
    model.eval()

    compressed_image_representations = []
    for batch_id, batch in enumerate(tqdm.tqdm(loader)):
        X, y = batch
        #y = y.to(device)
        output = model(X, setting="encode") #we are only passing the input through the encoder now to get the compressed representation
        compressed_image_representations.append(output.detach().cpu().numpy().reshape(output.size(0), -1)) # we need to flatten the representation

    compressed_image_representations = np.vstack(compressed_image_representations) #stacking the batches
    
    # CLUSTERING THE REPRESENTATIONS
    cluster_labels = {0: 'Abstract_Expressionism', 1: 'Action_painting', 2: 'Analytical_Cubism', 3: 'Art_Nouveau_Modern', 4: 'Baroque', 5: 'Color_Field_Painting', 6: 'Contemporary_Realism', 7: 'Cubism', 8: 'Early_Renaissance', 9: 'Expressionism', 10: 'Fauvism', 11: 'High_Renaissance', 12: 'Impressionism', 13: 'Mannerism_Late_Renaissance', 14: 'Minimalism', 15: 'Naive_Art_Primitivism', 16: 'New_Realism', 17: 'Northern_Renaissance', 18: 'Pointillism', 19: 'Pop_Art', 20: 'Post_Impressionism', 21: 'Realism', 22: 'Rococo', 23: 'Romanticism', 24: 'Symbolism', 25: 'Synthetic_Cubism', 26: 'Ukiyo_e'}
    
    num_clusters = 27 # we have 27 different art styles so ideally 27 clusters
    clustering_method = KMeans(n_clusters=num_clusters, random_state=42)
    clustering_method.fit(compressed_image_representations)
    cluster_labels_predict = clustering_method.predict(compressed_image_representations)
    cluster_centers = clustering_method.cluster_centers_ #needed for labeling the cluster centers, this should result in 27 cluster cetneres
    print(cluster_centers) 

    # DIMENSIONALITY REDUCTION FROM (1X300) IN ORDER TO PLOT THE REPRESENTATIONS
    pca = PCA(n_components=2)
    reduced_representations = pca.fit_transform(compressed_image_representations)
    reduced_centers = pca.transform(cluster_centers)

    # PLOTTING THE CLUSTERS
    plt.figure(figsize=(12, 12)) # tryin to make the plot larger to prevent datapoint overlapping
    scatter = plt.scatter(reduced_representations[:, 0], # x-cordinates
                          reduced_representations[:, 1], # y-cordinates
                          c=cluster_labels_predict,
                          cmap='gist_rainbow',
                         )
    plt.colorbar(scatter, label="Artstyle Label")  
    for i, center in enumerate(reduced_centers):
        x, y = center[0], center[1]
        label = cluster_labels[i]
        plt.text(x, y, label, horizontalalignment='center', verticalalignment='center')
    plt.title("WikiArt dataset art genre clusters (encoded representations)")
    plt.savefig("art_style_clusters_plotting.png") #saves the plot into current directory

    
test(modelfile=config["modelfile"], device=device)