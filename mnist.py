#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 22:13:50 2019

@author: alain
"""
"""Dataset setting and data loader for MNIST."""
"""https://github.com/corenel/pytorch-adda/tree/master/datasets"""

import torch
from torchvision import datasets, transforms


def get_mnist(train,batch_size = 32, drop_last=True,in_memory=True,
              num_channel=1,image_size=28):
    """Get MNIST dataset loader."""
    # image pre-processing   
    pre_process = transforms.Compose([transforms.Resize(image_size),
                                      transforms.Grayscale(num_channel),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=[0.5]*num_channel,
                                          std =[0.5]*num_channel)])

    # dataset and data loader
    mnist_dataset = datasets.MNIST(root='./',
                                   train=train,
                                   transform=pre_process,
                                   download=True)

    if in_memory == True:
        mnist_data_loader = torch.utils.data.DataLoader(
                dataset=mnist_dataset,
                batch_size= 1,
                shuffle=True,
                drop_last=False)
        data = torch.zeros((len(mnist_data_loader),num_channel,image_size,image_size))
        label = torch.zeros(len(mnist_data_loader))
        for i,(data_,target) in enumerate(mnist_data_loader):
            #print(i, data_.shape)
            data[i] = data_
            label[i] = target
        full_data = torch.utils.data.TensorDataset(data, label.long())
        mnist_data_loader = torch.utils.data.DataLoader(
                dataset=full_data,
                batch_size= batch_size,
                shuffle=True,
                drop_last=drop_last)
    else:
        mnist_data_loader = torch.utils.data.DataLoader(
                dataset=mnist_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True)

    return mnist_data_loader

if __name__ == '__main__':
    mnist_loader = get_mnist(train=True,num_channel=1,batch_size=1)
    data = torch.zeros((len(mnist_loader),1,28,28))
    label = torch.zeros(len(mnist_loader))
    for i,(data_,target) in enumerate(mnist_loader):
        print(i, data.shape[0])
        data[i] = data_[0,0]
        label[i] = target
    
    
