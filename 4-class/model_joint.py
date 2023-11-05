#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 20:23:01 2021

@author: sayan
"""

from collections import OrderedDict
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from convert_to_gpu import gpu
from convert_to_gpu_and_tensor import gpu_t
from convert_to_gpu_scalar import gpu_ts
from convert_to_cpu import cpu, cpu_ts


class joint_network(nn.Module):
    def __init__(self, gene_nodes, image_nodes, latent_dim):
        super(joint_network, self).__init__()

        gene_dict_en = OrderedDict()
        img_dict_en = OrderedDict()
        gene_dict_de = OrderedDict()
        img_dict_de = OrderedDict()

        for idx, node in enumerate(gene_nodes):
            if idx == len(gene_nodes) - 1:
                gene_dict_en["lin"+str(idx)] = nn.Linear(node, latent_dim, bias = False)
                break
            gene_dict_en['lin'+str(idx)] = nn.Linear(node, gene_nodes[idx + 1], bias=False)
            gene_dict_en['prelu'+str(idx)] = nn.PReLU()
            gene_dict_en['dropout'+str(idx)] = nn.Dropout(0.1)

        for idx, node in enumerate(image_nodes):
            if idx == len(image_nodes) -1:
                img_dict_en["lin"+str(idx)] = nn.Linear(node, latent_dim, bias = False)
                break
            img_dict_en['lin'+str(idx)] = nn.Linear(node, image_nodes[idx + 1], bias=False)
            img_dict_en['prelu'+str(idx)] = nn.PReLU()
            img_dict_en['dropout'+str(idx)] = nn.Dropout(0.1)
        
        for idx, node in enumerate(gene_nodes[::-1]):
            if idx == 0:
                gene_dict_de["lin"+str(idx)] = nn.Linear(latent_dim, node, bias = False)
                continue
            gene_dict_de['prelu'+str(idx)] = nn.PReLU()
            gene_dict_de['dropout'+str(idx)] = nn.Dropout(0.1)
            gene_dict_de['lin'+str(idx)] = nn.Linear(gene_nodes[::-1][idx - 1], node, bias=False)

        for idx, node in enumerate(image_nodes[::-1]):
            if idx == 0:
                img_dict_de["lin"+str(idx)] = nn.Linear(latent_dim, node, bias = False)
                continue
            img_dict_de['norm'+str(idx)] = nn.LayerNorm(image_nodes[::-1][idx - 1])
            img_dict_de['prelu'+str(idx)] = nn.PReLU()
            img_dict_de['dropout'+str(idx)] = nn.Dropout(0.1)
            img_dict_de['lin'+str(idx)] = nn.Linear(image_nodes[::-1][idx - 1], node, bias=False)
        
        print("Gene encoder:",gene_dict_en)
        print("Gene decoder:",gene_dict_de)
        print("Image encoder:",img_dict_en)
        print("Image decoder:",img_dict_de)

        self.encoder_g = nn.Sequential(gene_dict_en)

        self.decoder_g = nn.Sequential(gene_dict_de)

        self.combine = nn.Sequential(
            nn.PReLU(),
            nn.Dropout(0.1)
        )

        self.classification = nn.Sequential(
            nn.Linear(latent_dim, 25, bias=False),
            nn.LayerNorm(25),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(25, 1, bias=False),
        )

        self.encoder = nn.Sequential(img_dict_en)
        self.decoder = nn.Sequential(img_dict_de)
        self.bias_i = nn.ParameterList([nn.Parameter(0.1*(2*torch.rand(image_nodes[0], 2)-1))])
        self.prob = [0]
        self.bias_g  = nn.ParameterList([nn.Parameter(0.1*(2*torch.rand(gene_nodes[0],2)-1))])
        self.bias_ordinal = nn.Parameter(torch.zeros((1, 4)))

    def gumbel(self, alpha, t):
        u = (-torch.log(-torch.log(cpu(torch.rand(alpha.size())))) + alpha)/t
        return F.softmax(u, dim=1)

    def forward(self, x_g, x_i, T):
        surrogate_ig = []
        imp_i = cpu(torch.exp(self.bias_i[0]))
        imp_o_i = imp_i[:, 1]/torch.sum(imp_i, dim=1)
        self.imp = cpu(imp_o_i.detach()).data.numpy()

        imp_g = cpu(torch.exp(self.bias_g[0]))
        imp_o_g = imp_g[:, 1]/torch.sum(imp_g, dim=1)

        eps = cpu_ts(10**-6)
        if self.training:
            z_i = self.gumbel(torch.log(imp_i.repeat(x_i.size()[0], 1)+eps), T)
            bin_concrete_i = z_i[:, 1].reshape(x_i.size()[0], len(imp_o_i))

            z_g = self.gumbel(torch.log(imp_g.repeat(x_g.size()[0], 1)+eps), T)
            bin_concrete_g = z_g[:, 1].reshape(x_g.size()[0], len(imp_o_g))
            x_i_in = x_i*bin_concrete_i
            x_g_in = x_g*bin_concrete_g
        else:
            x_i_in = x_i.clone()
            x_g_in = x_g.clone()

        latent_g = self.encoder_g(x_g_in)

        latent_i = self.encoder(x_i_in)

        latent = (latent_g + latent_i)/2
        #latent = latent_g

        surrogate_ig.append(self.decoder_g(self.combine(latent)))

        surrogate_ig.append(self.decoder(self.combine(latent)))

        y_hat = self.classification(self.combine(latent))
        
        y_hat = F.sigmoid(y_hat.repeat(1,4) + self.bias_ordinal)

        prob = [imp_o_g, imp_o_i]
        self.prob = [prob[0].detach(), prob[1].detach()]
        return surrogate_ig, y_hat, prob
