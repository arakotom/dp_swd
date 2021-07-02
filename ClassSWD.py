#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import numpy as np
from time import time as tick 



import torch
import torch.nn as nn
import torch.optim as optim
from distrib_distance import sliced_wasserstein_distance, sliced_wasserstein_distance_diff_priv
from torch.autograd import grad
from sklearn.metrics import balanced_accuracy_score

def loop_iterable(iterable):
    while True:
        yield from iterable
def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad



def to_one_hot(labels,num_classes,cuda = False):
    labels = labels.reshape(-1, 1)
    if cuda:
        one_hot_target = (labels == torch.arange(num_classes).float())
    else:
        one_hot_target = (labels.cpu() == torch.arange(num_classes).float())
    return one_hot_target



#%%

def predict(data_loader,feat_extract,data_classifier,cuda):
    
    
    for i, (data,target) in enumerate(data_loader):
            if cuda:
                data = data.cuda()
            target = target.cpu().numpy()   
            
            output_feat = feat_extract(data)
            output = data_classifier(output_feat)
            pred = output.data.max(1)[1] # get the index of the max log-probability
    
            if i== 0:
                y_true = target
                y = pred.cpu().numpy()
                
            y_true = np.hstack((y_true,target))
            y = np.hstack((y,pred.cpu().numpy()))
    
    return y, y_true


def gradient_penalty(critic, h_s, h_t,cuda):
    # based on: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py#L116
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if cuda :
        device = 'cuda';
    else:
        device = 'cpu';
    #------------------------------------------------------------------------
    #alpha = torch.rand(h_s.size(0), 1).to(device)
    #differences = h_t - h_s
    #interpolates = (h_s + (alpha * differences))
    #interpolates = torch.stack([interpolates, h_s, h_t]).requires_grad_()
    #print(interpolates.shape)
    #-----------------------------------------------------------------------

    #interpolates = torch.cat((interpolates,h_s,h_t),dim=0).requires_grad_()
    #print(interpolates.shape)
    
    alpha = torch.rand(h_s.size(0), 1)
    alpha = (alpha.expand(h_s.size())).to(device)
    differences = h_t - h_s
    
    interpolates = (h_s + (alpha * differences))
    interpolates = torch.cat((interpolates,h_s,h_t),dim=0).requires_grad_()


    preds = critic(interpolates)
    gradients = grad(preds, interpolates,
                     grad_outputs=torch.ones_like(preds),
                     retain_graph=True, create_graph=True)[0]
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1)**2).mean()

    return gradient_penalty



class SWD(object):
    
    def __init__(self, feat_extractor,data_classifier, domain_classifier,source_data_loader, target_data_loader,
                 grad_scale = 1,cuda = False, logger_file = None, eval_data_loader = None, wgan = False, 
                 T_batches = None, S_batches = None):
        self.feat_extractor = feat_extractor
        self.data_classifier = data_classifier
        self.domain_classifier = domain_classifier
        self.source_data_loader = source_data_loader
        self.target_data_loader = target_data_loader

        self.eval_domain_data =0 # argument of list of eval_data_loader to use as domain evaluation with source
        self.eval_reference = 0
        self.source_domain_label = 1
        self.test_domain_label = 0
        self.cuda = cuda
        self.nb_iter = 1000
        self.logger = logger_file
        self.criterion = nn.CrossEntropyLoss()
        self.lr_decay_epoch = -1
        self.lr_decay_factor = 0.5
        self.wgan = wgan
        self.clamp = 0.1
        self.filesave = None
        self.save_best = True
        self.epoch_to_start_align  = 100 # start aligning distrib at this epoch
        self.iter_domain_classifier = 10
        self.T_batches = T_batches
        self.gamma = 10
        self.grad_scale_0 = grad_scale
        self.grad_scale = grad_scale
        self.compute_cluster_every= 10
        self.domain_classifier = domain_classifier
        self.num_projection = 1000
        
        # these are the default
        self.optimizer_feat_extractor = optim.SGD(self.feat_extractor.parameters(),lr = 0.001)
        self.optimizer_data_classifier = optim.SGD(self.data_classifier.parameters(),lr = 0.001)
        self.optimizer_domain_classifier = optim.SGD(self.domain_classifier.parameters(),lr = 0.01)

    def set_compute_cluster_every(self,compute):
        self.compute_cluster_every = compute
    def set_align_method(self,method):
        self.align_method = method
        print('align_method',method)
    def set_sigma_noise(self,noise):
        print('Sigma noise', noise)
        self.sigma_noise = noise
    def set_num_projection(self,num_projection):
        self.num_projection = num_projection
    def set_lr_decay_epoch(self,decay_epoch):
        self.lr_decay_epoch = decay_epoch
    def set_iter_domain_classifier(self,iter_domain_classifier):    
        self.iter_domain_classifier = iter_domain_classifier
    def set_epoch_to_start_align(self, epoch_to_start_align):
        self.epoch_to_start_align = epoch_to_start_align
        self.epoch_to_start_align_target = epoch_to_start_align

    def set_gamma(self,new_gamma):
        # regularizer for gradient penalty
        self.gamma = new_gamma
    #def set_prop(self,new_prop_factor):
    #    self.prop_factor = new_prop_factor
    def set_grad_scale(self,new_grad_scale):
           self.grad_scale = new_grad_scale
    #def set_cluster_param(self,new_cluster_param):
    #       self.cluster_param = new_cluster_param
    def set_filesave(self,filesave):
           self.filesave = filesave
    def show_grad_scale(self):
        print(self.grad_scale)
        return
    def set_n_class(self,n_class):
        self.n_class = n_class
        
    def set_optimizer_data_classifier(self, optimizer):
        self.optimizer_data_classifier = optimizer
    def set_optimizer_domain_classifier(self, optimizer):
        self.optimizer_domain_classifier = optimizer
    def set_optimizer_feat_extractor(self, optimizer):
        self.optimizer_feat_extractor = optimizer
    def set_nbiter(self, nb_iter):
        self.nb_iter = nb_iter
    def set_clamp(self,clamp_val):
        self.clamp = abs(clamp_val)
    def set_save_best(self,save_best):
        self.save_best = save_best
    def build_label_domain(self,size,label):
        label_domain = torch.LongTensor(size)       
        if self.cuda:
            label_domain = label_domain.cuda()
        
        label_domain.data.resize_(size).fill_(label)
        return label_domain
        
    def evaluate_data_classifier(self,data_loader, comments = ''):
        self.feat_extractor.eval()
        self.data_classifier.eval()
        
        test_loss = 0
        correct = 0
        y_pred = torch.Tensor()
        y_true = torch.zeros((0))
        for data, target in data_loader:
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            output_feat = self.feat_extractor(data)
            output = self.data_classifier(output_feat)
            test_loss += self.criterion(output, target).item()
            pred = output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()
            y_pred = torch.cat((y_pred,pred.float().cpu()))
            y_true = torch.cat((y_true,target.float().cpu()))
        MAP = balanced_accuracy_score(y_true,y_pred)  
        test_loss = test_loss
        test_loss /= len(data_loader) # loss function already averages over batch size  
        accur = correct.item() / (len(data_loader.dataset))

        print('{} Mean Loss:  {:.4f}, Accuracy: {}/{} ({:.0f}%) MAP :{:.4f}'.format(
                comments, test_loss, correct, len(data_loader.dataset),
                100*accur,MAP))
        
        if self.logger is not None:
            self.logger.info('{} Mean Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(comments, test_loss, correct, len(data_loader.dataset),
                accur))
        return accur,MAP
    def evaluate_domain_classifier_class(self, data_loader, domain_label):
        self.feat_extractor.eval()
        self.data_classifier.eval()
        self.grl_domain_classifier.eval()

        loss = 0
        correct = 0
        for data, _ in data_loader:
            target = self.build_label_domain(data.size(0),domain_label)
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            output_feat = self.feat_extractor(data)
            output = self.grl_domain_classifier(output_feat)
            loss += self.criterion(output, target).data[0]
            pred = output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()  
        return loss, correct
    
    def evaluate_domain_classifier(self):

        self.feat_extractor.eval()
        self.data_classifier.eval()
        self.grl_domain_classifier.eval()

        test_loss,correct = 0, 0
        test_loss, correct = self.evaluate_domain_classifier_class(self.source_data_loader, self.source_domain_label)
        loss, correct_a = self.evaluate_domain_classifier_class(self.eval_data_loader[self.eval_domain_data], self.test_domain_label)
        test_loss +=loss
        correct +=correct_a
        nb_source = len(self.source_data_loader.dataset) 
        nb_target = len(self. eval_data_loader[self.eval_domain_data].dataset) 
        nb_tot = nb_source + nb_target
        print('Domain: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                test_loss, correct, ( nb_source + nb_target ),
                100. * correct / (nb_source + nb_target )))
        if self.logger is not None:
             self.logger.info('Domain: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                test_loss, correct, ( nb_tot),
                100. * correct / nb_tot ))
        return correct / nb_tot 
    
    def fit(self):
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #print(device)
        if self.cuda:
            self.feat_extractor.cuda()
            self.data_classifier.cuda()
            self.domain_classifier.cuda()     
            device = 'cuda'
        else:
            device = 'cpu'

        k_critic = self.iter_domain_classifier
        gamma = self.gamma
        wd_clf = self.grad_scale
        boolean_normalization = True     
        maxi_norm = torch.Tensor([0.5]).to(device)

        for epoch in range(self.nb_iter):
            S_batches = loop_iterable(self.source_data_loader)

            
            batch_iterator = zip(S_batches, loop_iterable(self.target_data_loader))
            batch_iterator_wass = zip(S_batches, loop_iterable(self.target_data_loader))
            iterations = len(self.source_data_loader)
            total_loss = 0
            total_accuracy = 0
            tic = tick()

            for i in range(iterations):
                #print(i,iterations)
                (source_x, source_y), (target_x, _) = next(batch_iterator)
                # Train critic
                set_requires_grad(self.feat_extractor, requires_grad=False)
                set_requires_grad(self.domain_classifier, requires_grad=True)
            
                source_x, target_x = source_x.to(device), target_x.to(device)
                source_y = source_y.to(device)

                # ------------------------------------------------------------
                # Train classifier ad feature extractor 
                # once discriminator has been learned
                #-------------------------------------------------------------
                set_requires_grad(self.feat_extractor, requires_grad=True)
                set_requires_grad(self.domain_classifier, requires_grad=False)
                
                (source_x, source_y), (target_x, _) = next(batch_iterator)
                source_x, target_x = source_x.to(device), target_x.to(device)
                source_y = source_y.to(device)
                source_features = self.feat_extractor(source_x).view(source_x.shape[0], -1)
                target_features = self.feat_extractor(target_x).view(target_x.shape[0], -1)
        
                if epoch > self.epoch_to_start_align:
                    if self.align_method == 'WD':
                        wasserstein_distance = (self.domain_classifier(source_features)).mean() - self.domain_classifier(target_features).mean()       
                    elif self.align_method == 'SWD':
                        # plain sliced wasserstein
                        
                        wasserstein_distance = sliced_wasserstein_distance(source_features.view(source_features.shape[0], -1), 
                                                        target_features.view(target_features.shape[0], -1),
                                                        self.num_projection,2,
                                                        device)
                    
                    
                    elif self.align_method == 'SWDn':
                        if epoch > self.epoch_to_start_align:
                            #print('first')
                            with torch.no_grad():
                                maxi_norm = torch.sqrt(torch.max(torch.sum(target_features.view(target_features.shape[0], -1)**2,dim=1)))

                        source_features_norm = torch.div(source_features,(maxi_norm))
                        target_features_norm = torch.div(target_features,(maxi_norm))
                        wasserstein_distance = sliced_wasserstein_distance(source_features_norm.view(source_features.shape[0], -1), 
                                                        target_features.view(target_features_norm.shape[0], -1),
                                                        self.num_projection,2,
                                                        device)
                        wasserstein_distance = torch.mul(wasserstein_distance,maxi_norm)
                    elif self.align_method == 'SWD-DP':
                        if epoch > self.epoch_to_start_align:
                            with torch.no_grad():
                                maxi_norm = torch.sqrt(torch.max(torch.sum(target_features.view(target_features.shape[0], -1)**2,dim=1))).to(device)

                        source_features_norm = torch.div(source_features,(2*maxi_norm))
                        target_features_norm = torch.div(target_features,(2*maxi_norm))
                        wasserstein_distance = sliced_wasserstein_distance_diff_priv(target_features_norm.view(target_features.shape[0], -1),
                                                                                     source_features_norm.view(source_features.shape[0], -1), 
                                                        self.num_projection,2,
                                                        device,
                                                        sigma_noise=self.sigma_noise)
                        wasserstein_distance = torch.mul(wasserstein_distance,maxi_norm)

         
                    source_preds = self.data_classifier(source_features)                
                    self.criterion = nn.CrossEntropyLoss()
                    clf_loss = self.criterion(source_preds, source_y)
                    loss = clf_loss + wd_clf * wasserstein_distance   
                    #print(wasserstein_distance,clf_loss,maxi_norm)

                else:
                    wasserstein_distance = torch.zeros(1)
                    source_preds = self.data_classifier(source_features)
                    clf_loss = self.criterion(source_preds, source_y)
                    
                    
                    
                    loss = clf_loss
                self.optimizer_feat_extractor.zero_grad()
                self.optimizer_data_classifier.zero_grad()
                loss.backward()
                self.optimizer_feat_extractor.step()
                self.optimizer_data_classifier.step()
                total_accuracy +=clf_loss.item()
                    
                    
            toc =  tick() - tic 
            print('\n {} Train Epoch: {} {:2.2f}s \tLoss: {:.6f} DistLoss:{:.6f}'.format(
                        self.align_method,epoch, toc, total_accuracy, total_loss))
            self.evaluate_data_classifier(self.source_data_loader)
            self.evaluate_data_classifier(self.target_data_loader)


        
    def get_feature_extractor(self):
        return self.feat_extractor
    def get_data_classifier(self):
        return self.data_classifier
    
    def save_perf(self):
        np.savez(self.filesave + '.npz' ,accuracy_train = self.perf_source.numpy(), accuracy_evaluation = self.perf_val.numpy(),
                 accuracy_domain = self.perf_domain.numpy())
        


def exp_lr_scheduler(optimizer, epoch, lr_decay_epoch=100,lr_decay_factor=0.5):
    """Decay current learning rate by a factor of 0.5 every lr_decay_epoch epochs."""
    init_lr = optimizer.param_groups[0]['lr']
    if epoch > 0 and (epoch % lr_decay_epoch == 0):
        lr = init_lr*lr_decay_factor
        print('\n LR is set to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return optimizer

