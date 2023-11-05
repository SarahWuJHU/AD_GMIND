#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 13:50:10 2020

@author: sayan
"""
import torch
def cpu(x):
    x = x.to(torch.device("cpu"))
    return(x)

def cpu_t(x):
    y = torch.tensor(x.astype(float)).float()
    return(y)

def cpu_ts(x):
    y = torch.tensor(x).float()
    return(y)
    