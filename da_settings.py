#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 23:43:00 2019

@author: alain
"""

 


    

from usps import get_usps
from mnist import get_mnist
    
def expe_setting(setting,param):   
    opt = {}

    if setting < 10:
        opt['lr'] = 0.0005
        opt['grad_scale'] = 0.05



    
    
    if setting == 1:
        from da_models import DataClassifier, FeatureExtractor
        from da_models import DomainClassifier, DomainClassifierDANN

        batch_size = 128

        
        opt['source_loader'] = get_mnist(train=True,batch_size=batch_size)
        opt['target_loader'] = get_usps(train=True,batch_size=batch_size)
        opt['nb_iter'] = 20
        
        opt['feat_extract']= FeatureExtractor()
        opt['data_classifier'] = DataClassifier()
        opt['domain_classifier'] = DomainClassifier()
        opt['domain_classifier_dann'] = DomainClassifierDANN()
        
        opt['lr'] = 0.0005   
        opt['lr_dann'] = 0.001

        opt['grad_scale_dann'] = 0.05 
        opt['grad_scale'] = 0.05 
        opt['grad_scale_dp'] = 0.05 
        
        opt['nb_iter_alg'] = 100
        opt['iter_domain'] = 5
        opt['start_align'] = 10
        opt['batch_size'] = batch_size
        opt['nb_class'] = 10




        opt['sigma_noise_vec_swd'] = [1.07,0.2, 1, 2.76]
        opt['num_projections'] = 200
        
        opt['sigma_noise_vec_dann'] = 17

        filename = f"digits-setting{setting}"
        print('MNIST-USPS')
        return opt, filename
   
    if setting == 2:
        opt,filename = expe_setting(1, param)
        opt['sigma_noise_vec_swd'] = [1.02, 10, 15, 5.34]
        filename = f"digits-setting{setting}"
        opt['nb_iter'] = 10
        
        return opt,filename 
       
    #-------------------------------------------------------------------------
    #                       USPS-MNIST
    #-------------------------------------------------------------------------

    if setting == 10:  # balanced
        from da_models import DataClassifier, FeatureExtractor
        from da_models import DomainClassifier, DomainClassifierDANN
        batch_size = 128
        
        opt['source_loader'] = get_usps(train=True, batch_size=batch_size, drop_last=False)
        opt['target_loader'] = get_mnist(train=False, batch_size=batch_size, drop_last=False,
                                               num_channel=1)
        opt['nb_iter'] = 20
        opt['feat_extract']= FeatureExtractor()
        opt['data_classifier'] = DataClassifier()
        opt['domain_classifier'] = DomainClassifier()
        opt['domain_classifier_dann'] = DomainClassifierDANN()
        
        opt['lr'] = 0.005   
        opt['lr_dann'] = 0.005

        opt['grad_scale_dann'] = 0.01 # 
        opt['grad_scale'] = 0.01 #
        opt['grad_scale_dp'] = 1 # 
        
        opt['nb_iter_alg'] = 100
        opt['iter_domain'] = 5
        opt['start_align'] = 10
        opt['batch_size'] = batch_size
        opt['nb_class'] = 10
        
        opt['sigma_noise_vec_swd'] = [0.95,0.5, 1, 2.45]
        opt['num_projections'] = 200
        
        
        opt['sigma_noise_vec_dann'] = 14.65

        filename = f"digits-setting{setting}"
        print('USPS-MNIST')

        return opt, filename

    if setting == 11:
        opt,filename = expe_setting(10, param)
        opt['sigma_noise_vec_swd'] = [0.9, 6, 3, 4.74]
        filename = f"digits-setting{setting}"
        opt['nb_iter'] = 10
        
        return opt,filename

    
    if setting == 12:
        opt,filename = expe_setting(10, param)
        opt['sigma_noise_vec_swd'] = [10,15, 20, 25]
        filename = f"digits-setting{setting}"
        opt['nb_iter'] = 10
        
        return opt,filename    


    
    
    if setting == 20:
        from da_models import DataClassifierVisDA, FeatureExtractorVisDA
        from da_models import DomainClassifierVisDA, DomainClassifierDANNVisDA
        
        batch_size = 128


      
        opt['source_loader'] = get_visda(train=True,batch_size=batch_size,drop_last=False)
        opt['target_loader'] = get_visda(train=False,batch_size=batch_size,drop_last=False)
        
    
        opt['nb_iter'] = 20
        opt['feat_extract']= FeatureExtractorVisDA()
        opt['data_classifier'] = DataClassifierVisDA()
        opt['domain_classifier'] = DomainClassifierVisDA()
        opt['domain_classifier_dann'] = DomainClassifierDANNVisDA()
        
        opt['lr'] = 0.00005#0.0005    
        opt['lr_dann'] = 0.00005

        opt['grad_scale_dann'] = 0.1 # 0.05
        opt['grad_scale'] = 0.1 # 0.05
        opt['grad_scale_dp'] = 0.5 # 0.05
        
        opt['nb_iter_alg'] = 50
        opt['iter_domain'] = 5
        opt['start_align'] = 5
        opt['batch_size'] = batch_size
        opt['nb_class'] = 12
        opt['sigma_noise_vec_swd'] = [4, 2, 3, 2.441]
        opt['num_projections'] = 1000
        

        opt['sigma_noise_vec_dann'] = 4.75


        filename = f"visda-setting{setting}"

        return opt, filename
    if setting == 21:
        opt,filename = expe_setting(20, param)
        opt['sigma_noise_vec_swd'] = [2.32,10, 15, 6.48 ]
        filename = f"digits-setting{setting}"
        opt['nb_iter'] = 10
        opt['grad_scale_dp'] = 10 # 0.05
        opt['grad_scale'] = 10 # 0.05

        return opt,filename
    if setting == 22:
        opt,filename = expe_setting(20, param)
        opt['sigma_noise_vec_swd'] = [2.32,10, 15, 6.48 ]
        filename = f"digits-setting{setting}"
        opt['nb_iter'] = 10
        opt['grad_scale_dp'] = 10 # 0.05
        opt['grad_scale'] = 10 # 0.05

        return opt,filename


    if setting == 30:

        from da_models import DataClassifierHome, FeatureExtractorHome
        from da_models import DomainClassifierHome, DomainClassifierDANNHome

        source = 'dslr_dslr'
        target = 'dslr_webcam'
        n_class = 31
        print(source,target)

        batch_size = 32


        opt['source_loader'] = get_office31(source,batch_size=batch_size,drop_last=False)
        opt['target_loader'] = get_office31(target,batch_size=batch_size,drop_last=False)
        opt['nb_iter'] = 20
        opt['feat_extract']= FeatureExtractorHome()
        opt['data_classifier'] = DataClassifierHome(n_class=n_class)
        opt['domain_classifier'] = DomainClassifierHome()
        opt['domain_classifier_dann'] = DomainClassifierDANNHome()
        
        opt['lr'] = 0.0005#0.0005    
        opt['lr_dann'] = 0.0005

        opt['grad_scale_dann'] = 0.01 # 0.05
        opt['grad_scale'] = 0.1 # 0.05
        opt['grad_scale_dp'] = 0.5 # 0.05
        
        opt['nb_iter_alg'] = 50
        opt['iter_domain'] = 5
        opt['start_align'] = 10
        opt['batch_size'] = batch_size
        opt['nb_class'] = 31
        opt['sigma_noise_vec_swd'] = [3.245, 5, 12, 8.05]
        opt['num_projections'] = 100
        

        opt['sigma_noise_vec_dann'] = 9.81

        filename = f"office-setting{setting}"
        
        return opt,filename

    if setting == 31:

        from da_models import DataClassifierHome, FeatureExtractorHome
        from da_models import DomainClassifierHome, DomainClassifierDANNHome

        source = 'dslr_dslr'
        target = 'dslr_amazon'
        n_class = 31
        print(source,target)

        batch_size = 32


        opt['source_loader'] = get_office31(source,batch_size=batch_size,drop_last=False)
        opt['target_loader'] = get_office31(target,batch_size=batch_size,drop_last=False)
        opt['nb_iter'] = 20
        opt['feat_extract']= FeatureExtractorHome()
        opt['data_classifier'] = DataClassifierHome(n_class=n_class)
        opt['domain_classifier'] = DomainClassifierHome()
        opt['domain_classifier_dann'] = DomainClassifierDANNHome()
        
        opt['lr'] = 0.0005#0.0005    
        opt['lr_dann'] = 0.0005

        opt['grad_scale_dann'] = 1 # 0.05
        opt['grad_scale'] = 1 # 0.05
        opt['grad_scale_dp'] = 5 # 0.05
        
        opt['nb_iter_alg'] = 50
        opt['iter_domain'] = 5
        opt['start_align'] = 10
        opt['batch_size'] = batch_size
        opt['nb_class'] = 31
        opt['sigma_noise_vec_swd'] = [3.245, 5, 12, 8.05]
        opt['num_projections'] = 100
        

        opt['sigma_noise_vec_dann'] = 9.81

        filename = f"office-setting{setting}"
        
        return opt,filename

    if setting == 32:

        from da_models import DataClassifierHome, FeatureExtractorHome
        from da_models import DomainClassifierHome, DomainClassifierDANNHome

        source = 'amazon_amazon'
        target = 'amazon_webcam'
        n_class = 31

        batch_size = 32

        print(source,target)

        opt['source_loader'] = get_office31(source,batch_size=batch_size,drop_last=False)
        opt['target_loader'] = get_office31(target,batch_size=batch_size,drop_last=False)
        opt['nb_iter'] = 20
        opt['feat_extract']= FeatureExtractorHome()
        opt['data_classifier'] = DataClassifierHome(n_class=n_class)
        opt['domain_classifier'] = DomainClassifierHome()
        opt['domain_classifier_dann'] = DomainClassifierDANNHome()
        
        opt['lr'] = 0.00005#0.0005    
        opt['lr_dann'] = 0.00005

        opt['grad_scale_dann'] = 1 # 0.05
        opt['grad_scale'] = 0.1 # 0.05
        opt['grad_scale_dp'] = 5 # 0.05
        
        opt['nb_iter_alg'] = 50
        opt['iter_domain'] = 5
        opt['start_align'] = 10
        opt['batch_size'] = batch_size
        opt['nb_class'] = 31
        opt['sigma_noise_vec_swd'] = [3.245, 5, 12, 8.05]
        opt['num_projections'] = 100
        

        opt['sigma_noise_vec_dann'] = 9.81

        filename = f"office-setting{setting}"
        
        return opt,filename

    if setting == 33:

        from da_models import DataClassifierHome, FeatureExtractorHome
        from da_models import DomainClassifierHome, DomainClassifierDANNHome

        source = 'amazon_amazon'
        target = 'amazon_dslr'
        n_class = 31

        batch_size = 32
        print(source,target)


        opt['source_loader'] = get_office31(source,batch_size=batch_size,drop_last=False)
        opt['target_loader'] = get_office31(target,batch_size=batch_size,drop_last=False)
        opt['nb_iter'] = 20
        opt['feat_extract']= FeatureExtractorHome()
        opt['data_classifier'] = DataClassifierHome(n_class=n_class)
        opt['domain_classifier'] = DomainClassifierHome()
        opt['domain_classifier_dann'] = DomainClassifierDANNHome()
        
        opt['lr'] = 0.00005#0.0005    
        opt['lr_dann'] = 0.00005

        opt['grad_scale_dann'] = 1 # 0.05
        opt['grad_scale'] = 0.1 # 0.05
        opt['grad_scale_dp'] = 5 # 0.05
        
        opt['nb_iter_alg'] = 50
        opt['iter_domain'] = 5
        opt['start_align'] = 10
        opt['batch_size'] = batch_size
        opt['nb_class'] = 31
        opt['sigma_noise_vec_swd'] = [3.245, 5, 12, 8.05]
        opt['num_projections'] = 100
        

        opt['sigma_noise_vec_dann'] = 9.81

        filename = f"office-setting{setting}"
        
        return opt,filename
    
    if setting == 34:

        from da_models import DataClassifierHome, FeatureExtractorHome
        from da_models import DomainClassifierHome, DomainClassifierDANNHome

        source = 'webcam_webcam'
        target = 'webcam_amazon'
        n_class = 31

        batch_size = 32
        print(source,target)


        opt['source_loader'] = get_office31(source,batch_size=batch_size,drop_last=False)
        opt['target_loader'] = get_office31(target,batch_size=batch_size,drop_last=False)
        opt['nb_iter'] = 20
        opt['feat_extract']= FeatureExtractorHome()
        opt['data_classifier'] = DataClassifierHome(n_class=n_class)
        opt['domain_classifier'] = DomainClassifierHome()
        opt['domain_classifier_dann'] = DomainClassifierDANNHome()
        
        opt['lr'] = 0.00005#0.0005    
        opt['lr_dann'] = 0.0005

        opt['grad_scale_dann'] = 1 # 0.05
        opt['grad_scale'] = 0.1 # 0.05
        opt['grad_scale_dp'] = 5 # 0.05
        
        opt['nb_iter_alg'] = 50
        opt['iter_domain'] = 5
        opt['start_align'] = 10
        opt['batch_size'] = batch_size
        opt['nb_class'] = 31
        opt['sigma_noise_vec_swd'] = [3.245, 5, 12, 8.05]
        opt['num_projections'] = 100
        

        opt['sigma_noise_vec_dann'] = 9.81

        filename = f"office-setting{setting}"
        
        return opt,filename

    if setting == 35:

        from da_models import DataClassifierHome, FeatureExtractorHome
        from da_models import DomainClassifierHome, DomainClassifierDANNHome

        source = 'webcam_webcam'
        target = 'webcam_dslr'
        n_class = 31

        batch_size = 32
        print(source,target)

        opt['source_loader'] = get_office31(source,batch_size=batch_size,drop_last=False)
        opt['target_loader'] = get_office31(target,batch_size=batch_size,drop_last=False)
        opt['nb_iter'] = 20
        opt['feat_extract']= FeatureExtractorHome()
        opt['data_classifier'] = DataClassifierHome(n_class=n_class)
        opt['domain_classifier'] = DomainClassifierHome()
        opt['domain_classifier_dann'] = DomainClassifierDANNHome()
        
        opt['lr'] = 0.00005#0.0005    
        opt['lr_dann'] = 0.00005

        opt['grad_scale_dann'] = 0.1 # 0.05
        opt['grad_scale'] = 0.1 # 0.05
        opt['grad_scale_dp'] = 5 # 0.05
        
        opt['nb_iter_alg'] = 50
        opt['iter_domain'] = 5
        opt['start_align'] = 10
        opt['batch_size'] = batch_size
        opt['nb_class'] = 31
        opt['sigma_noise_vec_swd'] = [3.245, 5, 12, 8.05]
        opt['num_projections'] = 100
        

        opt['sigma_noise_vec_dann'] = 9.81

        filename = f"office-setting{setting}"
        
        return opt,filename
    if setting >= 40 and setting < 50:
        # version of 30-35 office without dp-dann
        opt,filename = expe_setting(setting-10, param)
        opt['sigma_noise_vec_swd'] = [3.245, 5, 12, 8.05]
        filename = f"office-setting{setting}"

        return opt,filename
    if setting >= 50 and setting < 60:
        # version of 30-35 office without dp-dann
        opt,filename = expe_setting(setting-20, param)
        opt['sigma_noise_vec_swd'] = [15, 20, 25, 11]
        filename = f"office-setting{setting}"

        return opt,filename
