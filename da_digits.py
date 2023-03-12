#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from time import process_time as time

import torch
import torch.optim as optim
import copy
from utils_local import  weight_init
from da_settings import expe_setting as expe_setting

# for USPS-MNIST
#       * choose setting 11, noise level can be chosen with noise_param
#       * noise_param = 3, corresponds to the noise for \varespilon=10, \delta=1e^-5 

# for MNIST-USPS
#       * choose setting 2, noise level can be chosen with noise_param
#       * noise_param = 3, corresponds to the noise for \varespilon=10, \delta=1e^-5 


setting = 11
noise_param = 3                              

param = 0
gpu = 0     
# 1 on the first bit for source, on second bit for DANN, on third for DPDANN
# on fourth for DP-SWD
modeltorun='0001'   

dtype = 'torch.DoubleTensor'
path_resultat = './resultat/digits/'


opt, filename = expe_setting(setting,param)
noise = opt['sigma_noise_vec_swd'][noise_param] 


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



filename = f"{filename}-{modeltorun}"
nb_iter = opt['nb_iter']
batch_size = opt['batch_size']

cuda = torch.cuda.is_available()
if cuda:
    gpu_id = gpu
    torch.cuda.set_device(gpu_id)
    print('cuda is ON')




bc_source = np.zeros(nb_iter)
bc_dann  = np.zeros(nb_iter)
bc_dpdann  = np.zeros(nb_iter)

bc_wdtsadv_clus  = np.zeros(nb_iter)

MAP_source = np.zeros(nb_iter)
MAP_dann  = np.zeros(nb_iter)
MAP_dpdann  = np.zeros(nb_iter)
MAP_wdtsadv_clus  = np.zeros(nb_iter)

bc_source_source = np.zeros(nb_iter)
bc_dann_source  = np.zeros(nb_iter)
bc_wdtsadv_clus_source  = np.zeros(nb_iter)

MAP_source_source = np.zeros(nb_iter)
MAP_dann_source  = np.zeros(nb_iter)
MAP_wdts_source  = np.zeros(nb_iter)
MAP_wdtsadv_source  = np.zeros(nb_iter)
MAP_wdtsadv_clus_source  = np.zeros(nb_iter)
MAP_wdtsadv_conf_source  = np.zeros(nb_iter)


for it in range(1):
    print(it)
    np.random.seed(it)
    torch.manual_seed(it)
    torch.cuda.manual_seed(it)


    source_loader = opt['source_loader']
    target_loader = opt['target_loader']

    
    feat_extract_init = opt['feat_extract']
    data_class_init = opt['data_classifier']
    domain_class_init = opt['domain_classifier']
    domain_class_dann_init = opt['domain_classifier_dann']
    
    feat_extract_init.apply(weight_init)
    data_class_init.apply(weight_init)
    domain_class_init.apply(weight_init)
    domain_class_dann_init.apply(weight_init)
    
    

    #%%
    # ------------------------------------------------------------------------
    # Source only
    # ------------------------------------------------------------------------
    if int(modeltorun[0])== 1:
        from ClassDann import DANN

        
        feat_extract_source = copy.deepcopy(feat_extract_init)
        data_class_source = copy.deepcopy(data_class_init)
        domain_class_source = copy.deepcopy(domain_class_dann_init)

        source = DANN(feat_extract_source, data_class_source,domain_class_source, source_loader,target_loader,batch_size,
                                  cuda = cuda)
        source.set_optimizer_feat_extractor(optim.Adam(source.feat_extractor.parameters(),lr=opt['lr_dann'],betas=(0.5, 0.999)))
        source.set_optimizer_data_classifier(optim.Adam(source.data_classifier.parameters(),lr=opt['lr_dann'],betas=(0.5, 0.999)))
        source.set_optimizer_domain_classifier(optim.Adam(source.grl_domain_classifier.parameters(),lr=opt['lr_dann'],betas=(0.5, 0.999)))
        source.set_nbiter(opt['nb_iter_alg'] )    

        source.set_grad_scale(opt['grad_scale_dann'])
        source.set_epoch_to_start_align(opt['nb_iter_alg'] )
        source.set_lr_decay_epoch(-1)
        source.fit()
        
    
    
        bc_source[it],MAP_source[it] = source.evaluate_data_classifier(target_loader)
        bc_source_source[it],MAP_source_source[it] = source.evaluate_data_classifier(source_loader)




    #%%
    # ------------------------------------------------------------------------
    # Domain adaptation with DANN
    # ------------------------------------------------------------------------
    if int(modeltorun[1])== 1:
        from ClassDann import DANN
        
        
        

        feat_extract_dann = copy.deepcopy(feat_extract_init)
        data_class_dann = copy.deepcopy(data_class_init)
        domain_class_dann = copy.deepcopy(domain_class_dann_init)

        dann = DANN(feat_extract_dann, data_class_dann,domain_class_dann, source_loader,target_loader,batch_size,
                                  cuda = cuda)
        dann.set_optimizer_feat_extractor(optim.Adam(dann.feat_extractor.parameters(),lr=opt['lr_dann'],betas=(0.5, 0.999)))
        dann.set_optimizer_data_classifier(optim.Adam(dann.data_classifier.parameters(),lr=opt['lr_dann'],betas=(0.5, 0.999)))
        dann.set_optimizer_domain_classifier(optim.Adam(dann.grl_domain_classifier.parameters(),lr=opt['lr_dann'],betas=(0.5, 0.999)))
        dann.set_nbiter(opt['nb_iter_alg'] )    
        dann.set_grad_scale(opt['grad_scale_dann'])
        dann.set_epoch_to_start_align(opt['start_align'])
        dann.set_lr_decay_epoch(-1)
        dann.fit()
        
    
    
        bc_dann[it],MAP_dann[it] = dann.evaluate_data_classifier(target_loader)

   #%%
    # ------------------------------------------------------------------------
    # Domain adaptation with DP - DANN
    # ------------------------------------------------------------------------
    if int(modeltorun[2])== 1:
        from ClassDann import DANN
        
        
        

        feat_extract_dann = copy.deepcopy(feat_extract_init)
        data_class_dann = copy.deepcopy(data_class_init)
        domain_class_dann = copy.deepcopy(domain_class_dann_init)

        dann = DANN(feat_extract_dann, data_class_dann,domain_class_dann, source_loader,target_loader,batch_size,
                                  cuda = cuda)
        dann.set_optimizer_feat_extractor(optim.Adam(dann.feat_extractor.parameters(),lr=opt['lr_dann'],betas=(0.5, 0.999)))
        dann.set_optimizer_data_classifier(optim.Adam(dann.data_classifier.parameters(),lr=opt['lr_dann'],betas=(0.5, 0.999)))
        dann.set_optimizer_domain_classifier(optim.Adam(dann.grl_domain_classifier.parameters(),lr=opt['lr_dann'],betas=(0.5, 0.999)))
        dann.set_nbiter(opt['nb_iter_alg'] )    
        dann.set_grad_scale(opt['grad_scale_dann'])
        dann.set_epoch_to_start_align(opt['start_align'])
        dann.set_lr_decay_epoch(-1)
        dann.set_dodp(True)
        dann.set_normclip(1)
        dann.set_sigma_noise(opt['sigma_noise_vec_dann'])
        dann.set_thresh_normclip(1)
        dann.fit()
        
    
    
        bc_dpdann[it],MAP_dpdann[it] = dann.evaluate_data_classifier(target_loader)
    # ------------------------------------------------------------------------
    # Domain adaptation with DP - SWD
    # ------------------------------------------------------------------------
      
    if int(modeltorun[3])== 1:
        do_wdtsadv = True
        from ClassSWD import SWD
        
        # create sub-networks
        feat_extract_wdtsadv = copy.deepcopy(feat_extract_init)
        data_class_wdtsadv =  copy.deepcopy(data_class_init)
        domain_class_wdtsadv = copy.deepcopy(domain_class_init)
    
        # compile model and fit
        dp_swd = SWD(feat_extract_wdtsadv, data_class_wdtsadv, domain_class_wdtsadv, source_loader,target_loader,
                                  cuda = cuda, grad_scale = 1 )
        dp_swd.set_optimizer_feat_extractor(optim.Adam(dp_swd.feat_extractor.parameters(),lr = opt['lr'],betas=(0.5, 0.999)))
        dp_swd.set_optimizer_data_classifier(optim.Adam(dp_swd.data_classifier.parameters(),lr =opt['lr'],betas=(0.5, 0.999)))
        dp_swd.set_optimizer_domain_classifier(optim.Adam(dp_swd.domain_classifier.parameters(),lr = opt['lr'],betas=(0.5, 0.999)))

        dp_swd.set_n_class(opt['nb_class'])
        dp_swd.set_nbiter(opt['nb_iter_alg'] )
        dp_swd.set_epoch_to_start_align(opt['start_align'])
        dp_swd.set_iter_domain_classifier(opt['iter_domain'])

        #----------------------------------------------------------------
        #  DP- SWP
        #          * choose SWD as align_method for non-private SWD
        #----------------------------------------------------------------
        dp_swd.set_align_method('SWD-DP')
        dp_swd.set_num_projection(opt['num_projections'])
        dp_swd.set_grad_scale(opt['grad_scale_dp']) 
        dp_swd.set_sigma_noise(noise) 


        tic = time()
        dp_swd.fit()
        time_wdtsadv_clus = time() - tic
        
        bc_wdtsadv_clus[it],MAP_wdtsadv_clus[it] = dp_swd.evaluate_data_classifier(target_loader)
        bc_wdtsadv_clus_source[it],MAP_wdtsadv_clus_source[it] = dp_swd.evaluate_data_classifier(source_loader)



#%%


