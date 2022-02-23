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
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import numpy as np


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
    def __init__(self, n_epochs = 50, lr = 0.01, method_='standard', p=0.5):
        self.n_epochs = n_epochs
        self.p_val = p
        self.dim = 28*28
        self.meth = method_
        
        self.train_load = data.DataLoader(
            datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor()), 
            batch_size=100, shuffle=True)
        
        self.test_load = data.DataLoader(
            datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor()), 
            batch_size=100, shuffle=False)
        
        self.net = Network(method=method_, p=self.p_val)
        self.loss = nn.CrossEntropyLoss()
        self.optim = optim.SGD(self.net.parameters(), lr=lr)
        self.ls_tr = []
        self.cr_tr = []
        #self.ls_ts = []
        #self.cr_ts = []
        
    def train(self):
        self.net.train()
        for epoch in range(self.n_epochs):
            epoch += 1
            total_loss = 0
            correct = 0
            len_ = 0
            for data_, label in self.train_load:
                data_, labels_ = Variable(data_).view(-1, self.dim), Variable(label)
                output = self.net(data_)
                _, predicted = torch.max(output.data, 1)
                loss_ = self.loss(output, labels_)
                self.optim.zero_grad()
                loss_.backward()
                self.optim.step()
                total_loss += float(loss_.data)
                correct += (predicted.cpu() == label).sum()
                len_ += label.size(0)
            total_loss /= len(self.train_load.dataset)
            cr = correct.item()/len_
            self.ls_tr.append(total_loss)
            self.cr_tr.append(cr)
            print('Epoch {} | loss: {:.6f}'.format(epoch, total_loss))
        self.plot_tool(self.ls_tr, self.cr_tr, self.p_val, self.meth, 'train')
            
    def test(self):
        self.net.eval()
        for epoch in range(self.n_epochs):
            epoch += 1
            correct = 0
            len_ = 0
            total_loss = 0
            for data_, label in self.test_load:
                data_, labels_ = Variable(data_).view(-1, self.dim), Variable(label)
                output = self.net(data_)
                _, predicted = torch.max(output.data, 1)
                loss_ = self.loss(output, labels_)
                correct += (predicted.cpu() == label).sum()
                total_loss += float(loss_.data)
                len_ += label.size(0)
            perc = 100*(correct.item()/len_)
            total_loss /= len(self.test_load.dataset)
            #cr = correct.item()/len_
            #self.ls_ts.append(total_loss)
            #self.cr_ts.append(cr)
            print('Accuracy: {:.4f}% | loss: {:.6f}'.format(perc, total_loss))
            
        #self.plot_tool(self.ls_ts, self.cr_ts, self.p_val, self.meth, 'test')
        
    def plot_tool(self, ls, cr, p_val, met, run):
        fig = plt.figure(figsize=(20, 10))
        plt.rc('font', family='serif')
        plt.xlabel('Epochs')
        
        ax = fig.gca()
        ax.xaxis.set_major_formatter(tick.FormatStrFormatter('%0g'))
        ax.xaxis.set_ticks(np.arange(0, self.n_epochs+5, 2))
        
        ax_sec = ax.twinx()
        ax.set_autoscaley_on(False)
        ax_sec.set_autoscaley_on(False)
        
        ax.plot(ls, linestyle='--', color='b', label='Loss') 
        ax_sec.plot(cr, linestyle='-', color='m', label='accuracy')
        ax.set_ylabel('Loss')
        ax_sec.set_ylabel('Accuracy')
        ax.set_ylim(min(ls), max(ls))
        ax_sec.set_ylim(min(cr), max(cr))
        ax.legend(loc="lower right")
        ax_sec.legend(loc='upper right')
        
        plt.title('P value: {}, method: {}, {}'.format(p_val, met, run))
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig('figures/Run_{}_pval_{}_{}.pdf'.format(run, p_val, met), format="pdf")
        plt.show(block=True)
        plt.close(fig)
        
## main function to run ##

for p_ in np.arange(0.0, 1.1, 0.3):
    standard = Train_Test(method_='standard', p=p_)
    standard.train()
    standard.test()
for p_ in np.arange(0.0, 1.1, 0.3):
    gauss = Train_Test(method_='gaussian', p=p_)
    gauss.train()
    gauss.test()
            
        