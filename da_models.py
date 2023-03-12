#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 22:12:29 2019

@author: alain
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
    

#dtype = 'torch.FloatTensor'
class DomainClassifierHome(nn.Module):
    def __init__(self):
        super(DomainClassifierHome, self).__init__()        
        self.fc1 = nn.Linear(50, 50,bias = True)     
        self.fc2 = nn.Linear(50, 1,bias = True)     

    def forward(self, input):        
        x = input
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
class DomainClassifierDANNHome(nn.Module):
    def __init__(self):
        super(DomainClassifierDANNHome, self).__init__()        
        self.fc1 = nn.Linear(50, 50,bias = True)     
        self.fc2 = nn.Linear(50, 2,bias = True)     

    def forward(self, input):        
        x = input
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class FeatureExtractorHome(nn.Module):

    def __init__(self):
        super(FeatureExtractorHome, self).__init__()        
        self.fc1 = nn.Linear(2048, 100,bias = True)     
        self.fc2 = nn.Linear(100, 50)     

    def forward(self, input):        
        x = input
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
class DataClassifierHome(nn.Module):
    def __init__(self,n_class=31):

        super(DataClassifierHome, self).__init__()
        #self.fc1 = nn.Linear(n_hidden, 2, bias = True)
        self.fc1 = nn.Linear(50, 50)
        self.fc2 = nn.Linear(50, n_class)
    def forward(self, input):

        x = input.view(input.size(0), -1)
        #x = self.fc1(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        #Fully Connected Layer/Activation
        x = self.fc2(x)
        return x




#------------------------------------------------------------------------------
#                   VISDA
#------------------------------------------------------------------------------
dtype = 'torch.FloatTensor'
class DomainClassifierVisDA(nn.Module):
    def __init__(self):
        super(DomainClassifierVisDA, self).__init__()        
        self.fc1 = nn.Linear(100, 100,bias = True)     
        self.fc2 = nn.Linear(100, 1,bias = True)     

    def forward(self, input):        
        x = input
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
class DomainClassifierDANNVisDA(nn.Module):
    def __init__(self):
        super(DomainClassifierDANNVisDA, self).__init__()        
        self.fc1 = nn.Linear(100, 100,bias = True)     
        self.fc2 = nn.Linear(100, 2,bias = True)     

    def forward(self, input):        
        x = input
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class FeatureExtractorVisDA(nn.Module):

    def __init__(self):
        super(FeatureExtractorVisDA, self).__init__()        
        self.fc1 = nn.Linear(2048, 100,bias = True)     
        self.fc2 = nn.Linear(100, 100)     

    def forward(self, input):        
        x = input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))  #
        return x
    
class DataClassifierVisDA(nn.Module):
    def __init__(self):

        super(DataClassifierVisDA, self).__init__()
        #self.fc1 = nn.Linear(n_hidden, 2, bias = True)
        self.fc1 = nn.Linear(100, 100)
        self.fc2 = nn.Linear(100, 12)
    def forward(self, input):

        x = input.view(input.size(0), -1)
        #x = self.fc1(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        #Fully Connected Layer/Activation
        x = self.fc2(x)
        return x




#------------------------------------------------------------------------------
#                   MNIST -- USPS
#------------------------------------------------------------------------------
dtype = 'torch.FloatTensor'
class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()        
        self.fc1 = nn.Linear(320, 100,bias = True)     
        self.fc2 = nn.Linear(100, 1,bias = True)     

    def forward(self, input):        
        x = input
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
class DomainClassifierDANN(nn.Module):
    def __init__(self):
        super(DomainClassifierDANN, self).__init__()        
        self.fc1 = nn.Linear(320, 100,bias = True)     
        self.fc2 = nn.Linear(100, 2,bias = True)     

    def forward(self, input):        
        x = input
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()        
        #self.fc1 = nn.Linear(dim, n_hidden, bias = True)
        #self.fc2 = nn.Linear(n_hidden, n_hidden, bias = True)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()  #Dropout
    def forward(self, input):
        x = F.relu(F.max_pool2d(self.conv1(input), 2)) 
        #Convolutional Layer/Dropout/Pooling Layer/Activation
        x = torch.sigmoid(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        
        
        x = x.view(-1, 320)
        #x = torch.div(x,320)
        #x = F.relu(self.fc1(input))
        #x = F.relu(self.fc2(x))
        return x 

class DataClassifier(nn.Module):
    def __init__(self):

        super(DataClassifier, self).__init__()
        #self.fc1 = nn.Linear(n_hidden, 2, bias = True)
        self.fc1 = nn.Linear(320, 100)
        self.fc2 = nn.Linear(100, 10)
    def forward(self, input):

        x = input.view(input.size(0), -1)
        #x = self.fc1(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        #Fully Connected Layer/Activation
        x = self.fc2(x)
        return x
#%%%
#------------------------------------------------------------------------------
#                   USPS -- MNIST
#------------------------------------------------------------------------------

class FeatureExtractorDigits(nn.Module):
    def __init__(self, channel, kernel_size=5):
        super(FeatureExtractorDigits, self).__init__()
        self.conv1 = nn.Conv2d(channel, 64, kernel_size=kernel_size)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=kernel_size)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 64 * 2, kernel_size=kernel_size)
        self.bn3 = nn.BatchNorm2d(64 * 2)

    def forward(self, input):
        x = self.bn1(self.conv1(input))
        x = self.relu1(self.pool1(x))
        x = self.bn2(self.conv2(x))
        x = self.relu2(self.pool2(x))
        x = torch.sigmoid(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return x


class DataClassifierDigits(nn.Module):
    def __init__(self, n_class):
        super(DataClassifierDigits, self).__init__()
        input_size = 64 * 2

        self.fc1 = nn.Linear(input_size, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.relu1 = nn.ReLU()
        self.dp1 = nn.Dropout2d()
        self.fc2 = nn.Linear(100, 100)
        self.bn2 = nn.BatchNorm1d(100)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(100, n_class)

    def forward(self, input):
        x = self.dp1(self.relu1(self.bn1(self.fc1(input))))
        x = self.relu2(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


class DomainClassifierDANNDigits(nn.Module):
    def __init__(self, bigger_discrim=True):
        super(DomainClassifierDANNDigits, self).__init__()
        self.domain_classifier = nn.Sequential()
        input_size = 64 * 2
        output_size = 500 if bigger_discrim else 100

        self.bigger_discrim = bigger_discrim
        self.fc1 = nn.Linear(input_size, output_size)
        # self.bn1 = nn.BatchNorm1d(output_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(output_size, 100) if bigger_discrim else nn.Linear(output_size, 2)
        # self.bn2 = nn.BatchNorm1d(100)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(100, 2)

    def forward(self, input):
        x = self.relu1(self.fc1(input))  # self.relu1(self.bn1(self.fc1(input)))
        if self.bigger_discrim:
            x = self.relu2(self.fc2(x))  # self.relu2(self.bn2(self.fc2(x)))
            x = self.fc3(x)
        else:
            x = self.fc2(x)
        return x


class DomainClassifierDigits(nn.Module):
    def __init__(self, bigger_discrim=True):
        super(DomainClassifierDigits, self).__init__()
        self.domain_classifier = nn.Sequential()
        input_size = 64 * 2
        output_size = 500 if bigger_discrim else 100

        self.bigger_discrim = bigger_discrim
        self.fc1 = nn.Linear(input_size, output_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(output_size, 100) if bigger_discrim else nn.Linear(output_size, 2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(100, 1)

    def forward(self, input):
        x = self.relu1(self.fc1(input))
        if self.bigger_discrim:
            x = self.relu2(self.fc2(x))
            x = self.fc3(x)
        else:
            x = self.fc2(x)
        return x
    
    
    




if __name__== '__main__':

    feat = FeatureExtractor()
    classifier = DataClassifier()
    print(feat(torch.randn(15,1,28,28)).shape)
    
    #aux = feat(torch.randn(15,3,32,32))
    #output = classifier(aux)
    domain=DomainClassifier()
    print(domain(torch.randn(15,320)).shape)
