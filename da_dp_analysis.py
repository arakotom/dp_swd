#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 23:30:31 2021

@author: alain


"""

from autodp import rdp_acct, rdp_bank
import numpy as np
from scipy.stats import norm







#%%% Gradient Clipping DA
setting = 0

if setting == 1:
    print("\n \t\t--- MNIST-USPS---\n")
elif setting == 0:
    print("\n \t\t--- USPS-MNIST---\n")
elif setting == 2:
    print("\n \t\t---VisDA---\n")
elif setting == 3:
    print("\n \t\t--- Office31---\n")

delta = 1e-5

if setting== 0:   # USPS-MNIST
    N = 10000
    batch_size = 128
    epoch = 100
    sigma = 14.65

elif setting == 1: #MNIST-USPS
    N = 7438
    batch_size = 128
    epoch = 100
    sigma = 17
elif setting == 2: # Visda
    delta = 1e-6

    N = 55387
    batch_size = 128
    epoch = 50
    sigma = 4.75
elif setting == 3:
    d =50
    delta = 1e-3
    batch_size = 32
    N = 497
    epoch = 50   
    sigma = 9.81
    
prob = batch_size/N  # subsampling rate
n_steps = epoch/prob*batch_size # training iterations the multiplication of batchisize
                                # is due to the composition of the mechanism on all
                                # elements of the batch

acct = rdp_acct.anaRDPacct()
func = lambda x: rdp_bank.RDP_gaussian({'sigma': sigma}, x)

acct = rdp_acct.anaRDPacct()
acct.compose_subsampled_mechanism(func, prob, coeff=n_steps)
epsilon = acct.get_eps(delta)
print("DP-DANN : epsilon={:2.3f}, delta={:2.3e} sigma={:2.3f}".format(epsilon, delta,sigma))

 

#%%  DP -SWD Setting and privacy accountant

if setting == 0:  # USPS-MNIST
    d = (28*1)**2
    batch_size = 128
    N = 10000
    nb_epoch = 100  
    
    num_projections = 400
    sigma_of_noise = 1.27 #the noise that you add'
    
    num_projections = 200        
    clt = False
    if clt:
        sigma_of_noise = 0.9 #the noise that you add'
    else:
        # using bernstein ine
        sigma_of_noise = 4.74
    
elif setting == 1: # MNIST - USPS
    d = (28*1)**2
    batch_size = 128
    N = 7468
    nb_epoch = 100   
    
    num_projections = 200
    #sigma_of_noise = 0.65#the noise that you add

    clt = True
    if clt:
        sigma_of_noise = 1.02#the noise that you add'
    else:
        # using bernstein one
        sigma_of_noise = 5.34 #the noise that you add'
    
elif setting == 2: # VisDA
    d = 100
    delta = 1e-6
    batch_size = 128
    N = 55387
    nb_epoch = 50   
    
    num_projections = 1000
    #sigma_of_noise = 0.65#the noise that you add

    clt = True
    if clt:
        sigma_of_noise = 2.32#the noise that you add'
    else:
        sigma_of_noise = 6.48 #the noise that you add'

elif setting == 3: # Office 31
    d =50
    delta = 1e-3
    batch_size = 32
    N = 497  # Calibrating noise level on the worst case
    nb_epoch = 50   
    num_projections = 100
    clt = False
    if clt:
        sigma_of_noise = 3.245#the noise that you add'
    else:
        sigma_of_noise = 8.05 #the noise that you add'
a = 1/2
b = (d - 1)/2 
delta2 = delta/2
v_est = a*b/((a+b)**2*(a+b+1))*num_projections

if clt:
    # CLT Bound
    z = norm.isf(delta2,0,1)
    sensitivity_est = np.sqrt(num_projections/d+  z/d *np.sqrt(2*num_projections*(d-1)/(d+2)))    
else:
    # Bernstein Bound
    sensitivity_est = np.sqrt(num_projections/d+ 2/3*np.log(1/delta2) + np.sqrt(2*num_projections*v_est*np.log(1/delta2)))




sigma = sigma_of_noise/sensitivity_est # sigma of DP 
batch_per_epoch = N//batch_size+1
coeff = batch_per_epoch * 5



func = lambda x: rdp_bank.RDP_gaussian({'sigma': sigma}, x)
k= nb_epoch*batch_per_epoch
prob = batch_size/N
acct = rdp_acct.anaRDPacct()
acct.compose_subsampled_mechanism(func, prob, coeff=batch_per_epoch*nb_epoch)
epsilon = acct.get_eps(delta)
print("DP-SWD : epsilon={:2.3f}, delta={:2.3e} sigma={:2.3f}".format(epsilon, delta,sigma_of_noise))


