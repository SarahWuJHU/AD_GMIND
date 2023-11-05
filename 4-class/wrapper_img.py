#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 16:08:19 2019

@author: sayan
"""
import sys, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib
import scipy.io as sio
from scipy.io import loadmat
import shutil
import os
from net_train_joint import net_train_joint
from model_joint  import joint_network
from convert_to_gpu import gpu
from convert_to_gpu_and_tensor import gpu_t
from convert_to_gpu_scalar import gpu_ts
from convert_to_cpu import cpu, cpu_t, cpu_ts
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
import collections
pwd = os.getcwd()

############################################################################################################
# necessary defintions

def initialize_check_losses(check_losses,s,num, test_n):
    for i in s:
        if i=='class_pred' or i=='class_true':
            check_losses[i] = -np.ones((num,test_n))
        else:
            check_losses[i] = np.zeros((num))
    return 

def helper_test(net, x_g_j, x_i_j, y_j, D, typ, hyp, e, metric, T):
    net.eval()
    with torch.no_grad():            
        # when all modalities are present
        loss_j, loss1_j, loss2_j, loss4_j, y_j_hat = helper(net, x_g_j.detach(), x_i_j.detach(), y_j.detach(), 'j', criterion_recon, criterion_class, hyp, T)

        D[typ]['gene_recon_loss'][e]    = loss1_j  
        D[typ]['image_recon_loss'][e]   = loss2_j  
        D[typ]['class_loss'][e]         = loss4_j  
        D[typ]['total_loss'][e]         = loss_j 
        
        Y     = cpu(y_j).data.numpy().sum(axis = 1)
        Y_hat = (cpu(y_j_hat).data.numpy()> ordinal_thres).cumprod(axis = 1).sum(axis = 1)
    
        y_label = Y
        y_predict = Y_hat
        metric[typ]['F1'][e] = metrics.f1_score(y_label, y_predict, average='macro')
        metric[typ]['Acc'][e] = metrics.accuracy_score(y_label, y_predict)
        metric[typ]['Kappa'][e] = metrics.cohen_kappa_score(y_label, y_predict)
        

        D[typ]['class_true'][e,:] = Y
        D[typ]['class_pred'][e,:] = Y_hat
        
    return
        
def helper_plot(M, e, f, st):
    for i in M:
        if i!='class_pred' and i!='class_true':
            plt.plot(M[i][0:e].squeeze().tolist())
            plt.savefig(file_folder +'data_'+str(id_fold)+'/model'+str(f)+'/'+st+'_'+i+'.pdf')
            plt.close()  
    return

def train_helper_plot(M, e, f, st):
    for i in M:
        plt.plot(M[i][0:e])
        plt.savefig(file_folder +'data_'+str(id_fold)+'/model'+str(f)+'/'+st+'_'+i+'.pdf')
        plt.close()
      
def helper(net, x_g, x_i, y, mode, criterion_recon, criterion_class, hyp, T):
    net.eval()
    with torch.no_grad():
        #tresholding the importance score with prob_g
        x_g_t = x_g.clone().detach()*torch.unsqueeze(net.prob[0].clone().detach()>0.1, 0)
        #x_g_t = x_g.clone().detach()*torch.unsqueeze(net.prob[0].clone().detach()>prob_g,0)
        x_i_t = x_i.clone().detach()*torch.unsqueeze(net.prob[1].clone().detach()>0.1, 0)
        
        surrogate_ig, y_hat, _ = net(x_g_t, x_i_t, T)

        loss1 = hyp[0]*torch.sum(criterion_recon(surrogate_ig[0], x_g))
        loss2 = hyp[1]*torch.sum(criterion_recon(surrogate_ig[1], x_i)) 
        loss3 = hyp[2]*torch.sum(criterion_class(y_hat, y))
        loss = loss1 + loss2 + loss3
        
    return cpu(loss), cpu(loss1), cpu(loss2), cpu(loss3), cpu(y_hat)

def save_results(net, metric_losses, check_losses, train_losses, f, prob_d):
    vis = {}
    
    for i in train_losses:
        vis['train'+i] = train_losses[i]
        
    for i in check_losses['test']:
        vis['test'+i] = check_losses['test'][i]
        vis['val' +i] = check_losses['val'][i]
        
    for i in metric_losses['test']:
        vis['train'+i] = metric_losses['test'][i]
        vis['test'+i] = metric_losses['test'][i]
        vis['val' +i]  = metric_losses['val'][i]
    
    vis['prob_gene'] = prob_d['gene']    
    vis['prob_Image'] = prob_d['Image']
    
    sio.savemat(file_folder +'data_'+str(id_fold)+'/setting/model'+str(f)+'/vis'+'.mat',vis)   
    
    
    

#####################################################################################################
'''
id_fold = int(sys.argv[1])
'''
id = 2
#plt.ioff()
 
##################################################################################################################################################################################

# Loading data.
# /home/sarah/Desktop/GMIND/
# /home/swu82/data-avenka14/sarah
# training data
id_fold = id 
pwd = '/home/sarah/Desktop/GMIND/4_class/'
I_train = cpu_t(loadmat(pwd + 'cross_val/fold' + str(id_fold) + '.mat')['I_train'])
G_train = cpu_t(loadmat(pwd + 'cross_val/fold' + str(id_fold) + '.mat')['G_train'])
Y_train = cpu_t(loadmat(pwd + 'cross_val/fold' + str(id_fold) + '.mat')['Y_train_O'])

## Validation data
I_val = cpu_t(loadmat(pwd + 'cross_val/fold' + str(id_fold) + '.mat')['I_val'])
G_val = cpu_t(loadmat(pwd + 'cross_val/fold' + str(id_fold) + '.mat')['G_val'])
Y_val = cpu_t(loadmat(pwd + 'cross_val/fold' + str(id_fold) + '.mat')['Y_val_O'])

# test data
I_test = cpu_t(loadmat(pwd + 'cross_val/fold' + str(id_fold) + '.mat')['I_test'])
G_test = cpu_t(loadmat(pwd + 'cross_val/fold' + str(id_fold) + '.mat')['G_test'])
Y_test = cpu_t(loadmat(pwd + 'cross_val/fold' + str(id_fold) + '.mat')['Y_test_O'])

test_n = Y_test.size()[0]
val_n  = Y_val.size()[0]

############################################################################################################################################################################
e_losses = []
num_epochs = 1000
num_s = 0
num_f = 100

iter_n = 21

# Thresholdin probabilities.
#sps_G = [0.1, 0.3, 0.6]
#sps_N = [0.1, 0.3, 0.6]  

ordinal_thresholds = [0.5, 0.6, 0.7, 0.8]
ordinal_thres = 0.5

# Batch Size
batchs = [8, 32, 64, 128]
batch_size = 32

# Learning rate
multipilier = [10, 5, 1, 0.1, 0.01]
learning_scale = 5
gene_layer = [1165, 10]
img_layer = [68, 128, 64]


# cost weights : lambda
#choose so that they are in a similar nu
# merical range
lambda_0 = [cpu_ts(0.001), cpu_ts(0.01),cpu_ts(1), cpu_ts(0.001)] # gene_recon, image_recon, class loss, sprasity

# sprsity probability
prob_ref = [cpu_ts(0.0001), cpu_ts(0.0001)]

# gumbell parameter
temperature = cpu_ts(0.1)

l_dim = 10

#warm start iteration numbers
iter_ths = 0

# Define the loss functions
# multiclass classification - need to change loss function  you have already done softmax 
criterion_class = nn.BCELoss(reduction='none')
criterion_recon = nn.MSELoss(reduction='none') 

# Create folder to store models and results
file_folder = 'batch' + str(batch_size) + 'rate' + str(learning_scale) + '/'
if( not os.path.exists(file_folder) ):
    os.mkdir(file_folder) # Create cross validation folder f.

if( not os.path.exists(file_folder + 'data_'+str(id_fold)) ):
    os.mkdir(file_folder + 'data_'+str(id_fold)) # Create cross validation folder f.

if( not os.path.exists(file_folder + 'data_'+str(id_fold)+'/setting') ):
    os.mkdir(file_folder + 'data_'+str(id_fold)+'/setting') # Create cross validation folder f.
    
  
for f in range(id_fold, id_fold+1):
    
    if( os.path.exists(file_folder + 'data_'+str(id_fold)+'/model'+str(f)) ):
        shutil.rmtree(file_folder + 'data_'+str(id_fold)+'/model'+str(f))
        
    print('fold - '+str(f))
    #shutil.rmtree('model'+str(f))
    os.mkdir(file_folder + 'data_'+str(id_fold)+'/model'+str(f)) # Create cross validation folder f.
    if  os.path.exists(file_folder + 'data_'+str(id_fold)+'/setting/model'+str(f)) :
        shutil.rmtree(file_folder + 'data_'+str(id_fold)+'/setting/model'+str(f))
        
    os.mkdir(file_folder + 'data_'+str(id_fold)+'/setting/model'+str(f)) # Create cross valida+ 0.1*loss2tion folder f.
    
    # Couple modules
    net =  cpu(joint_network(gene_layer, img_layer, l_dim))
    
    # init optimizer
    layers = [{'params':net.bias_ordinal},{'params': net.encoder.parameters()}, {'params': net.decoder.parameters()}, \
        {'params': net.encoder_g.parameters(), 'lr':0.00002*learning_scale}, {'params': net.decoder_g.parameters(), 'lr':0.00002*learning_scale}, \
        {'params': net.combine.parameters()}, {'params': net.classification.parameters()},\
        {'params': net.bias_i.parameters(), 'lr':0.0001*learning_scale}, {'params':net.bias_g.parameters(), 'lr':0.0005*learning_scale}]
    #layers = net.parameters()
    opt_j       = optim.Adam(layers, lr=0.00005*learning_scale, betas=(0.9, 0.999), weight_decay=0)

    scheduler_j = torch.optim.lr_scheduler.StepLR(opt_j, step_size=500, gamma = 0.5) # this will decrease the learning rate by factor of 0.1
    
    # initialize losses losses
    train_losses = {'total_loss':[], 'gene_recon_loss':[], 'image_recon_loss':[], 'class_loss':[],'sparsity_loss':[]}
    check_losses = collections.defaultdict(dict)
    l = ['total_loss', 'gene_recon_loss', 'image_recon_loss', 'class_loss', 'class_pred', 'class_true']
    check_losses['val']   = {}
    check_losses['test']  = {}
    initialize_check_losses(check_losses['val'] , l, num_epochs, val_n)
    initialize_check_losses(check_losses['test'], l, num_epochs, test_n)
    
    metric_losses = collections.defaultdict(dict)
    l = ['F1', 'Acc', 'Kappa']
    metric_losses['train']   = {}
    metric_losses['test']    = {}
    metric_losses['val']     = {}
    initialize_check_losses(metric_losses['train'], l, num_epochs, test_n)
    initialize_check_losses(metric_losses['val']  , l, num_epochs, val_n)
    initialize_check_losses(metric_losses['test'] , l, num_epochs, test_n)
    
    prob_data = {}
    prob_data['gene'] = np.zeros((num_epochs,1165))
    prob_data['Image'] = np.zeros((num_epochs,68))
    
    for e in range(num_epochs):
        print(e)
        
        l_j,   l_g_j,   l_i_j , l_c_j,  l_s_j, y_j_hat, y_j  = net_train_joint(net, G_train, I_train, Y_train, \
                                                                                                    opt_j, batch_size, temperature, lambda_0, prob_ref, criterion_class, \
                                                                                                    criterion_recon, e, iter_n, iter_ths)
        
        scheduler_j.step()
        
        
        # store train losses
        train_losses['total_loss'].append(l_j )
        train_losses['image_recon_loss'].append(l_i_j )
        train_losses['gene_recon_loss'].append(l_g_j )
        train_losses['class_loss'].append(l_c_j )
        train_losses['sparsity_loss'].append(l_s_j)  
        
        Y_tr     = cpu(y_j).data.numpy().sum(axis = 1)
        Y_hat_tr = (cpu(y_j_hat).data.numpy()> ordinal_thres).cumprod(axis = 1).sum(axis = 1)
    
        y_label = Y_tr
        y_predict = Y_hat_tr
        metric_losses['train']['F1'][e] = metrics.f1_score(y_label, y_predict, average='macro')
        metric_losses['train']['Acc'][e] = metrics.accuracy_score(y_label, y_predict)
        metric_losses['train']['Kappa'][e] = metrics.cohen_kappa_score(y_label, y_predict)
        
        # testing and validation
        net.eval()
        with torch.no_grad():
            prob_data_g = cpu(net.prob[0]).data.numpy()
            prob_data_i = cpu(net.prob[1]).data.numpy()
            prob_data['gene'][e,:] = prob_data_g
            prob_data['Image'][e,:] = prob_data_i
            helper_test(net, G_test, I_test, Y_test, \
                         check_losses, 'test', lambda_0, e, metric_losses, temperature)
            
            helper_test(net, G_val, I_val, Y_val, \
                        check_losses, 'val', lambda_0, e, metric_losses, temperature)
            
            if e%10==0:
                save_results(net, metric_losses, check_losses, train_losses, f, prob_data)
        
        if e%num_f == 0 :
            
            helper_plot( metric_losses['test'], e, f, 'test')
            helper_plot( check_losses['test'],  e, f, 'test')
            helper_plot( metric_losses['val'], e, f, 'val')
            helper_plot( check_losses['val'],  e, f, 'val')
            train_helper_plot( train_losses, e, f, 'train')
            
            gen_pred = cpu(net.prob[0]).data.numpy()
            plt.stem(list(range(np.shape(gen_pred)[0])),gen_pred)
            plt.savefig(file_folder +'data_'+str(id_fold)+'/model'+str(f)+'/prob_pred_Gen.pdf')
            plt.close()
            
            gen_pred = cpu(net.prob[1]).data.numpy()
            plt.stem(list(range(np.shape(gen_pred)[0])),gen_pred)
            plt.savefig(file_folder +'data_'+str(id_fold)+'/model'+str(f)+'/prob_pred_Nback.pdf')
            plt.close()
                
            
            
        
