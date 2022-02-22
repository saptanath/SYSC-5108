#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 11:09:12 2022

@author: saptaparnanath
"""

import torch 
import torch.nn as nn
import torch.utils.data as data
from torchvision import datasets, transforms
import torch.optim as optim
from torch.autograd import Variable


class Gaussian(nn.Module):
    def __init__(self, p_val):
        super(Gaussian, self).__init__()
        self.mean = 1.0
        self.std = p_val/(1-p_val)
        
    def forward(self, data):
        if self.train():
            epsilon = torch.normal(self.mean, self.std, size=data.size())
            epsilon = Variable(epsilon)
            return data*epsilon
        else:
            return data

class Network(nn.Module):
    def __init__(self, image_dim = 28*28, method='standard', p=0.5):
        super(Network, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(image_dim, 2048),
            self.dropmethod(p, method), 
            nn.ReLU(), 
            nn.Linear(2048, 2048),
            self.dropmethod(p, method),
            nn.ReLU(), 
            nn.Linear(2048, 10))
        
    def forward(self, inp):
        return self.network(inp)
        
    def dropmethod(self, p=None, method='standard'):
        if method == 'standard':
            return nn.Dropout(p)
        else:
            return Gaussian(p)
            
        
    
class Train_Test(object):
    def __init__(self, n_epochs = 100, lr = 0.01, method_='standard', p=0.5):
        self.n_epochs = n_epochs
        self.p_val = p
        self.dim = 28*28
        
        self.train_load = data.DataLoader(
            datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor()), 
            batch_size=100, shuffle=True)
        
        self.test_load = data.DataLoader(
            datasets.MNIST('../data', train=False, download=True, transform=transforms.ToTensor()), 
            batch_size=1000, shuffle=False)
        
        self.net = Network(method=method_, p=self.p_val)
        self.loss = nn.CrossEntropyLoss()
        self.optim = optim.SGD(self.net.parameters(), lr=lr)
        
    def train(self):
        self.net.train()
        for epoch in range(self.n_epochs):
            epoch += 1
            total_loss = 0
            for data_, label in self.train_load:
                data_, label = Variable(data_).view(-1, self.dim), Variable(label)
                output = self.net(data_)
                loss_ = self.loss(output, label)
                self.optim.zero_grad()
                loss_.backward()
                self.optim.step()
                total_loss += float(loss_.data)
            total_loss /= len(self.train_load.dataset)
            print('Epoch {} | loss: {:.6f}'.format(epoch, total_loss))
            
    def test(self):
        self.net.eval()
        correct = 0
        len_ = 0
        for data_, label in enumerate(self.test_load):
            data_, label = Variable(data_).view(-1, self.dim), Variable(label)
            output = self.net(data_)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted.cpu() == label).sum()
            len_ += label.size(0)
        perc = 100*(correct/len_)
        print('Accuracy: {:.4f}%'.format(perc))
        
standard = Train_Test(method_='gaussian')
standard.train()
standard.test()
            
        