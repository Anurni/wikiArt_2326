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
    """
    Model for creating compressed (latent) representations of the images in the dataset.
    The compressing of the inputs is achieved by setting the stride parameter to 2.
    The same stride will allow us to reconstruct the images in the decoding phase.
    """
    def __init__(self, latent_compressed_rep_dim=300):
        super().__init__()
        
        # CONVOLUTING (encoding) LAYERS ARE DEFINED HERE:
        ######################################
        self.encoder = nn.Sequential(
        # we start with image input dimensions of (3, 416*416) 
        nn.Conv2d(3, 16, 2, stride=2, padding=1), # n of in channels = 3, n of out channels=16, 2x2 kernel and stride of 2. 
        #After the first conv layer, the output dimensions will be (16, 209*209) since our stride is 2 and the goal is to compress.
        nn.ReLU(),
        nn.Conv2d(16, 32, 2, stride=2, padding=1), # n of in channels = 16, n of out channels=32, 2x2 kernel and stride of 2.
        #After the second conv layer, the output dimensions will be (32, 105*105) since our stride is 2 and the goal is to compress.
        nn.ReLU(),
        nn.Conv2d(32, 64, 2, stride=2, padding=1), #n of in channels=32, n of out channels=64, 2x2 kerkel and stride of 2.
        #After the third conv layer, the output dimensions will be (64, 53*53) since our stride is 2 and the goal is to compress.
        nn.ReLU(),
        nn.Flatten(),
        # compressed representation layer, resulting in a vector with dimension of 1x300, these will be used for clustering
        nn.Linear((64*53*53), latent_compressed_rep_dim)
        )

        # DECONVOLUTING (decoding) LAYERS ARE DEFINED HERE:
        ########################################
        self.decoder = nn.Sequential(
        # compressed rep into back into decoder input size
        nn.Linear(latent_compressed_rep_dim, 64*53*53),
        # we start with the input image dimensions from self.decoder_input layer, which has 64 channels and 2809 pixels
        nn.ReLU(),
        nn.Unflatten(1, (64, 53, 53)), 
        nn.ConvTranspose2d(64, 32, 2, stride=2, padding=1, output_padding=1),
        #After the first deconvoluting layer, the output dimensions will be (32,105*105) since stride is 2 and the goal is to decompress.
        nn.ReLU(),
        nn.ConvTranspose2d(32, 16, 2, stride=2, padding=1, output_padding=1),
        #After the second deconvoluting layer, the output dimensions will be (16,209*209) since stride is 2 and the goal is to decompress.
        nn.ReLU(),
        nn.ConvTranspose2d(16, 3, 2, stride=2, padding=1),
        #After the third deconvoluting layer, the output dimensions will be (3, 416*416) since stride is 2 ad the goal is to decompress.
        nn.ReLU()
        )
        
    

    def forward(self, image, setting="both"):
        """
        The forward pass of the model. 
        By specifying the setting parameter as both the user can choose to only encode the inputs.
        """
        if setting=="encode":
            return self.encoder(image)
        else:
            encoded = self.encoder(image)
            decoded = self.decoder(encoded)
            return decoded


