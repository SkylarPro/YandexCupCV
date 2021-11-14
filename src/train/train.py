#!/usr/bin/env python
# coding: utf-8

# In[131]:


import sys
import os

# sys.path.append("path/to/YandexCup")
from load_model import load_model
from CustomDataset import Sent2textDataset,sempler

import torch.nn as nn
import numpy as np
import torch
from torch.optim import Adam,lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from typing import List,Tuple
from tqdm import tqdm

config_train = {
    "weights": "V3GPTsmall_new.pt",
    "lr": 1e-03,
    "weight_decay": 1e-05,
    "log_dir": "runs",
    "T_0" : 10,
    "T_mult": 2,
    "eta_min": 1e-09,
    "n_epoch": 5,   
    "img":1, # коэф перед лоссом 
    "text":1,# коэф перед лоссом 
    "clust":1, # коэф перед лоссом 
    "split_data":0.2,
    "bs":8,
    "nfreeze_layers_thead":16,
    "nfreeze_layers_ihead":8,
    "p_save_model":"model/weight"
}

config_data = {
    "path_t_csv": "data/Concat_N1-7.csv",
    "path_i_json": "data/images.json",
    "path_i_folders": ["data/images7","data/images6","data/images5",
                 "data/images4","data/images3","data/images2","data/images1"],
    "down_data": False,
    "path_to_load":"",
    "check_img": False,
    "len_seq": 15,
    "mode":"Sber",
    "resize_img":224,
    
}


class Train_model:
    
    class Args:
        def __init__(self,cfg):
            for k,v in cfg.items():
                setattr(self,k,v)
    
    def __init__(self,config_train = None, config_data = None):
        self.args = self.Args(config_train)
        
        self.dataset = Sent2textDataset(config_data)
        self.model, _, _ = load_model(path_to_model=self.args.weights)
        if not os.path.isdir(self.args.p_save_model):
            os.makedirs(self.args.p_save_model)
        
        self.train_dl, self.valid_dl = sempler(self.dataset, batch_size=self.args.bs, split=self.args.split_data)
       
        #utils for train
        self.loss = nn.CrossEntropyLoss()
        self._freeze_layers_model("visual")
        self._freeze_layers_model("text")
        self.optimizer = Adam([param for param in self.model.parameters() if param.requires_grad],
                              lr = self.args.lr, weight_decay = self.args.weight_decay
                             )
        self.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,
                                                      T_0=self.args.T_0, 
                                                      T_mult=self.args.T_mult,
                                                      eta_min=self.args.eta_min)
        
        if self.args.log_dir:
            self.writer = SummaryWriter(self.args.log_dir)
            #
    
    def _freeze_layers_model(self,head):
        print(f"Freeze head {head}")
        if head == "visual":
            n_trans = 0
            for n_module, modul in enumerate(self.model.visual_encoder.model.children()):
                if n_module + n_trans >= self.args.nfreeze_layers_ihead:
                    return
                if n_module == 2:
                    for id_trans, transfomres in enumerate(modul.resblocks.children()):
                        print(f"Id {n_trans + n_module} Freeze {transfomres}")
                        for param in transfomres.parameters():
                            param.requires_grad = False
                        if id_trans + n_module >= self.args.nfreeze_layers_ihead:
                            return
                        n_trans = id_trans
                else:
                    print(f"Id {n_trans + n_module} Freeze {modul}")
                    for param in modul.parameters():
                        param.requires_grad = False
                    
        if head == "text":
            n_trans = 0
            for n_module, modul in enumerate(self.model.text_encoder.model.children()):
                if n_module + n_trans >= self.args.nfreeze_layers_thead:
                    return
                if n_module == 3:
                    for id_trans, transfomres in enumerate(modul.children()):
                        print(f"Id {n_trans + n_module}Freeze {transfomres}")
                        for param in transfomres.parameters():
                            param.requires_grad = False
                        if id_trans + n_module >= self.args.nfreeze_layers_thead:
                            return
                        n_trans = id_trans
                else:
                    print(f"Id {n_trans + n_module} Freeze {modul}")
                    for param in modul.parameters():
                        param.requires_grad = False
                        
    
    def plot_train_params(self, params:List[Tuple[str,Tuple[int,int]]]):
        for param in params:
            self.writer.add_scalar(param[0],param[1][0],param[1][1])
    
    def train_model(self,):
        loss_history = []
        train_history = []
        val_history = []
        val_loss_hist = []
        metric_y_val = metric_p_val = None
        
        self.model.cuda()
                
        for epoch in range(self.args.n_epoch):
            
            print(epoch)
            self.model.train()
            
            correct_samples = 0
            correct_samples_cls = 0
            
            total_samples_cls = 0
            total_samples = 0
            
            loss_accum = 0
            loss_accum_cls = 0
            loss_accim_i2t = 0
            
            params_plot = []
            for i_step, data  in tqdm(enumerate(self.train_dl)):
                
                
                    imgs_gpu = torch.squeeze(data[0].cuda(),1)
                    texts_gpu = torch.squeeze(data[1].cuda(),1)
                    att_mask_gpu = data[2].cuda()
                    label_gpu = torch.arange(data[0].shape[0]).cuda()
                    cls_gpu = data[3].cuda()
                    
                    
                    prediction, per_img, cls = self.model(img_input={"x": imgs_gpu},
                                        text_input={"text": texts_gpu, "attention_mask":att_mask_gpu})
                    
                    
                    loss_valuei = self.loss(per_img, label_gpu)*self.args.img
                    loss_valuet = self.loss(prediction, label_gpu)*self.args.text
                    loss_cls = self.loss(cls,cls_gpu.reshape(-1))*self.args.clust
                    
                    loss_value =  (loss_cls + loss_valuet + loss_valuei) / 3
                    
                    _, preds = torch.max(prediction, 1)
                    _, preds_cls = torch.max(cls, 1)
                    
                    self.optimizer.zero_grad()
                    loss_value.backward()
                    self.optimizer.step()
                    
                    if i_step == 0 and epoch == 0:
                        metric_y = label_gpu.cpu().numpy()
                        metric_p = preds.cpu().numpy()
                    else:
                        metric_y = np.concatenate((metric_y, label_gpu.cpu().numpy()))
                        metric_p = np.concatenate((metric_p, preds.cpu().numpy())) 
                        
                    correct_samples += torch.sum(preds == label_gpu)
                    correct_samples_cls += torch.sum(preds_cls == cls_gpu.reshape(-1))
                    
                    loss_accum += loss_value
                    loss_accum_cls += loss_cls
                    loss_accim_i2t += loss_valuet
                    
                    total_samples += label_gpu.shape[0]
                    total_samples_cls += cls_gpu.shape[0]
                    
                    del imgs_gpu
                    del texts_gpu
                    del att_mask_gpu
                
                    del label_gpu
                    del cls_gpu
                            
            ave_loss = loss_accum / (i_step + 1)
            ave_loss_cls = loss_accum_cls / (i_step + 1)
            ave_loss_i2t = loss_accim_i2t / (i_step + 1)
            
            
            torch.save(self.model.state_dict(),f"{self.args.p_save_model}/last.pt")
            if len(loss_history) > 0 and min(loss_history) >= ave_loss:
                torch.save(self.model.state_dict(),f"{self.args.p_save_model}/best.pt")
            
            loss_history.append(ave_loss)
            
            train_accuracy = correct_samples / total_samples
            train_accuracy_cls = correct_samples_cls / total_samples_cls
            
            params_plot.append(("Loss/train",(ave_loss,epoch)))
            params_plot.append(("Loss_cls/train",(ave_loss_cls,epoch)))
            params_plot.append(("Loss_i2t/train",(ave_loss_i2t,epoch)))
            
            params_plot.append(("Acc/train",(train_accuracy,epoch)))
            params_plot.append(("Acc_cls/train",(train_accuracy_cls,epoch)))
            
            self.plot_train_params(params_plot, writer)
            
            print(f"Train Average loss: {ave_loss}, Average cls loss: {ave_loss_cls}, Average i2t loss: {ave_loss_i2t},  Accuracy it2: {train_accuracy}, Accuracy cls: {train_accuracy_cls}")
            
            metric_y_val, metric_p_val = self.compute_valid(epoch,metric_y_val, metric_p_val)
            self.plot_train_params([("Lr/epoch", (scheduler.get_last_lr()[-1], epoch))])
            self.scheduler.step(epoch)
            
            
            
            print('Train Epoch:', epoch, 'LR:', scheduler.get_last_lr())
            
        self.writer.close()
        return model
    
    def compute_valid(self,epoch,metric_y = None ,metric_p = None):
        self.model.cuda()
        self.model.eval()
        with torch.no_grad():
            
            total_samples_cls = 0
            total_samples = 0
            
            correct_samples = 0
            correct_samples_cls = 0
            
            loss_accum = 0
            loss_accum_cls = 0
            loss_accim_i2t = 0
            
            params = []
            for i_step, data in tqdm(enumerate(self.valid_dl)):
                
                imgs_gpu = torch.squeeze(data[0].cuda(),1)
                texts_gpu = torch.squeeze(data[1].cuda(),1)
                att_mask_gpu = data[2].cuda()
                label_gpu = torch.arange(data[0].shape[0]).cuda()
                cls_gpu = data[3].cuda()
    
                prediction, per_img,cls = self.model(img_input={"x": imgs_gpu},
                                        text_input={"text": texts_gpu, "attention_mask":att_mask_gpu})
                
                loss_cls = self.loss(cls,cls_gpu.reshape(-1))*self.args.clust
                loss_valuei = self.loss(per_img, label_gpu)*self.args.img
                loss_valuet = self.loss(prediction, label_gpu)*self.args.text
                
                loss_value = (loss_valuet + loss_cls + loss_valuei) / 3
                
                
                _, preds = torch.max(prediction, 1)
                _, preds_cls = torch.max(cls, 1)
                
                if i_step == 0 and epoch == 0:
                    metric_y = label_gpu.cpu().numpy()
                    metric_p = preds.cpu().numpy()
                else:
                    metric_y = np.concatenate((metric_y, label_gpu.cpu().numpy()))
                    metric_p = np.concatenate((metric_p, preds.cpu().numpy())) 
                
                
                correct_samples += torch.sum(preds == label_gpu)
                correct_samples_cls += torch.sum(preds_cls == cls_gpu.reshape(-1))
                
                
                total_samples += label_gpu.shape[0]
                total_samples_cls += cls_gpu.shape[0]
                
                loss_accum += loss_value
                loss_accum_cls +=loss_cls
                loss_accim_i2t += loss_valuet
    
                del imgs_gpu
                del texts_gpu
                del att_mask_gpu
                
                del label_gpu
                del cls_gpu
                    
            ave_loss_val = loss_accum / (i_step + 1)
            ave_loss_val_cls = loss_accum_cls / (i_step + 1)
            ave_loss_val_i2t = loss_accim_i2t / (i_step + 1)
            
            val_accuracy = correct_samples / total_samples
            val_accuracy_cls = correct_samples_cls / total_samples_cls
            
            params.append(("Loss/valid",(ave_loss_val, epoch)))
            params.append(("Loss_cls/valid",(ave_loss_val_cls, epoch)))
            params.append(("Loss_i2t/valid",(ave_loss_val_i2t, epoch)))
            
            params.append(("Acc/valid",(val_accuracy, epoch)))
            params.append(("Acc_cls/valid",(val_accuracy_cls, epoch)))
            
            self.plot_train_params(params, writer)
            
            print(f"Val Average loss: {ave_loss_val}, Average cls loss: {ave_loss_val_cls}, Average i2t loss: {ave_loss_val_i2t},  Accuracy it2: {val_accuracy}, Accuracy cls: {val_accuracy_cls}")
            
            return metric_y, metric_p
        
if __name__ == "__main__":
    procces = Train_model(config_train,config_data)
    procces.train_model()




