"""Dataset setting and data loader for USPS.

Modified from

https://github.com/corenel/pytorch-adda/
"""

import gzip
import os
import pickle
import urllib

import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets, transforms

#import params


class USPS(data.Dataset):
    """USPS Dataset.

    Args:
        root (string): Root directory of dataset where dataset file exist.
        train (bool, optional): If True, resample from dataset randomly.
        download (bool, optional): If true, downloads the dataset
            from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that takes in
            an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
    """

    url = "https://raw.githubusercontent.com/mingyuliutw/CoGAN/master/cogan_pytorch/data/uspssample/usps_28x28.pkl"

    def __init__(self, root, train=True, transform=None, download=True):
        """Init USPS dataset."""
        # init params
        self.root = os.path.expanduser(root)
        self.filename = "usps_28x28.pkl"
        self.train = train
        # Num of Train = 7438, Num ot Test 1860
        self.transform = transform
        self.dataset_size = None

        # download dataset.
        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You can use download=True to download it")

        self.train_data, self.train_labels = self.load_samples()
        if self.train:
            total_num_samples = self.train_labels.shape[0]
            indices = np.arange(total_num_samples)
            np.random.shuffle(indices)
            self.train_data = self.train_data[indices[0:self.dataset_size], ::]
            self.train_labels = self.train_labels[indices[0:self.dataset_size]]
        #self.train_data *= 255.0
        self.train_data = self.train_data.transpose(
            (0, 2, 3, 1))  # convert to HWC
        print(self.train_labels.shape)
        self.train_labels = self.train_labels

    def __getitem__(self, index):
        """Get images and target for data loader.

        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, label = self.train_data[index, ::], self.train_labels[index]
        #print(label)
        if self.transform is not None:
            img = self.transform(img)
        #label = torch.LongTensor([np.int64(label).item()])
        label = int(label)
        return img, label

    def __len__(self):
        """Return size of dataset."""
        return self.dataset_size

    def _check_exists(self):
        """Check if dataset is download and in right place."""
        return os.path.exists(os.path.join(self.root, self.filename))

    def download(self):
        """Download dataset."""
        filename = os.path.join(self.root, self.filename)
        dirname = os.path.dirname(filename)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        if os.path.isfile(filename):
            return
        print("Download %s to %s" % (self.url, os.path.abspath(filename)))
        urllib.request.urlretrieve(self.url, filename)
        print("[DONE]")
        return

    def load_samples(self):
        """Load sample images from dataset."""
        filename = os.path.join(self.root, self.filename)
        f = gzip.open(filename, "rb")
        data_set = pickle.load(f, encoding="bytes")
        f.close()
        if self.train:
            images = data_set[0][0]
            labels = data_set[0][1]
            self.dataset_size = labels.shape[0]
        else:
            images = data_set[1][0]
            labels = data_set[1][1]
            self.dataset_size = labels.shape[0]
        return images, labels

def get_usps(train,batch_size = 32, drop_last=True,in_memory=True):
    """Get USPS dataset loader."""
    # image pre-processing
    image_size = 28
    pre_process = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(
                                          #mean=params.dataset_mean,
                                          #std=params.dataset_std)]
                                          mean=(0.5,),
                                          std =(0.5,))])

    # dataset and data loader
    usps_dataset = USPS(root= './',
                        train=train,
                        transform=pre_process,
                        download=True)
    if in_memory == True:
        usps_data_loader = torch.utils.data.DataLoader(
                dataset=usps_dataset,
                batch_size= 1,
                shuffle=True,
                drop_last=False)
        data = torch.zeros((len(usps_data_loader),1,image_size,image_size))
        label = torch.zeros(len(usps_data_loader))
        for i,(data_,target) in enumerate(usps_data_loader):
            #print(i, data_.shape[0])
            data[i] = data_[0,0]
            label[i] = target
        full_data = torch.utils.data.TensorDataset(data, label.long())
        usps_data_loader = torch.utils.data.DataLoader(
                dataset=full_data,
                batch_size= batch_size,
                shuffle=True,
                drop_last=drop_last)
    
    else:
        usps_data_loader = torch.utils.data.DataLoader(
                dataset=usps_dataset,
                batch_size= batch_size,
                shuffle=True,
                drop_last=drop_last)
    

    return usps_data_loader
if __name__ == '__main__':
    usps_loader = get_usps(train=True,batch_size=1,drop_last=False,in_memory=True)
    data = torch.zeros((len(usps_loader),1,28,28))
    label = torch.zeros(len(usps_loader))
    for i,(data_,target) in enumerate(usps_loader):
        print(i, data.shape[0])
        data[i] = data_[0,0]
        label[i] = target
    import matplotlib.pyplot as plt
    
    
    
    
