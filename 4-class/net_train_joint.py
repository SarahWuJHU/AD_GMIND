#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 13:07:38 2020

@author: sayan
"""
import torch
import torch.nn as nn
import torch.nn.functional as Fun
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data.sampler import BatchSampler
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from convert_to_gpu import gpu
from convert_to_gpu_and_tensor import gpu_t
from convert_to_cpu import cpu, cpu_ts
from convert_to_gpu_scalar import gpu_ts

def  net_train_joint(net, x_g, x_i, Y, opt, batch_size, temperature, lambda0, prob_ref, criterion_class, criterion_recon, e, iter_n, iter_ths):
    
    net.train()
    
    eps = cpu_ts(1e-10)

    y_labels = np.sum(cpu(Y).data.numpy(), axis = 1)
    class_sample_count = np.array([len(np.where(y_labels == t)[0]) for t in np.unique(y_labels)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[int(t-1)] for t in y_labels])
    samples_weight = torch.from_numpy(samples_weight)
    w_sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight), replacement=False)
    sampler = BatchSampler(w_sampler, batch_size=batch_size, drop_last=False)
    
    X_G          =   x_g[:,:]
    X_I          =   x_i[:,:] #if (mode=='j' or mode=='N') else torch.tensor(0).float() #else you freeze it
    Y            =   Y[:,:]

    
    # initialize losses
    losses               = []
    losses_sparsity      = []
    losses_class         = []
    losses_gene_recon    = []
    losses_img_recon     = []
    
    # training metic 
    y_true = cpu(torch.tensor([]))
    y_pred = cpu(torch.tensor([]))
    
    for beg in sampler:
        # Initialise input.
        X_G_batch         = X_G[beg, :]
        X_I_batch         = X_I[beg, :]
        y_batch           = Y[beg, :]

        opt.zero_grad()
        
        # sets gradient for all parameters
        for param in net.parameters():
            param.requires_grad = True
        
        # freeze gene encoder for couple of iterations.         
        for param in net.bias_i.parameters():
            param.requires_grad = True

        noise = torch.randn(X_I_batch.size())*0.1
        surrogate_ig, y_hat, prob = net(X_G_batch, X_I_batch + noise, temperature)
        
        s2 = cpu_ts(0)
        
        for i in range(len(prob)):
            #KL divergence with prob_ref
            rho = cpu(torch.FloatTensor([prob_ref[i] for _ in range(prob[i].size()[0])]))        
            rho_hat = prob[i]

            #KL divergence
            x1 = rho #Fun.softmax(rho,dim=1)          
            x2 = rho_hat
            
            #ensure  the probability is sparse
            s1  = torch.sum(x2 * (torch.log(x2+eps) - torch.log(x1+eps)))
            if i == 1:
                s2 += (torch.sum((1 - x2) * (torch.log(1 - x2+eps) - torch.log(1 - x1+eps))) + s1) * (X_G_batch.size()[1]/X_I_batch.size()[1])
            else: 
                s2 += (torch.sum((1 - x2) * (torch.log(1 - x2+eps) - torch.log(1 - x1+eps))) + s1)
            
        gene_recon_loss   = lambda0[0]*torch.sum(criterion_recon(surrogate_ig[0], X_G_batch))
        
        image_recon_loss  = lambda0[1]*torch.sum(criterion_recon(surrogate_ig[1], X_I_batch))
        
        class_loss        = lambda0[2]*torch.sum(criterion_class(y_hat, y_batch))
        
        sparsity_loss     = lambda0[3]*s2
        
        loss = (gene_recon_loss + image_recon_loss + class_loss)/batch_size + sparsity_loss/9
        
        #(3) Compute gradients
        loss.backward()
        
        # (4) update weights
        opt.step() 
        
        # compile losses
        losses.append(cpu(loss.detach()).data.numpy())
        losses_gene_recon.append(cpu(gene_recon_loss.detach()).data.numpy()/batch_size)
        losses_img_recon.append(cpu(image_recon_loss.detach()).data.numpy()/batch_size)
        losses_sparsity.append(cpu(sparsity_loss.detach()).data.numpy()/9)
        losses_class.append(cpu(class_loss.detach()).data.numpy()/batch_size)
        
        # compile predictions
        y_pred = torch.cat((y_pred,y_hat.detach()))
        y_true = torch.cat((y_true,y_batch.detach()))
        
    # mean across batches 
    ll       = np.mean(losses)
    ll_g     = np.mean(losses_gene_recon)
    ll_i     = np.mean(losses_img_recon)
    ll_class = np.mean(losses_class)
    ll_reg   = np.mean(losses_sparsity)
    
    return ll, ll_g, ll_i, ll_class, ll_reg, y_pred, y_true
     
