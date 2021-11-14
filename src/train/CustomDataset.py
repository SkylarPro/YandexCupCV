#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import sys

# sys.path.append("path/to/YandexCup")
from os import listdir
import os
sys.path.append(os.getcwd())
from os.path import isfile, join

import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import SubsetRandomSampler

import cv2
from PIL import Image


import pandas as pd
from typing import List, Tuple, Dict

import multiprocessing as mp
import requests
import jsonlines

from tqdm import tqdm
import time
import random


from clip.origin.clip import _transform
from clip.evaluate.utils import  (
    get_text_batch, get_image_batch,get_tokenizer, get_text_batch_BPE
)
from load_model import get_tokenizer


data_config = {
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


def sempler(data_train, batch_size = 4, split = .2):
    
    data_size = len(data_train)

    validation_split = split
    split = int(np.floor(validation_split * data_size))
    indices = list(range(data_size))
    np.random.shuffle(indices)

    train_indices,val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size,
                                              sampler=train_sampler,)
    
    val_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size,
                                            sampler=val_sampler,)

    return train_loader,val_loader



class Sent2textDataset(Dataset):
    class Args:
        def __init__(self,cfg):
            for k,v in cfg.items():
                setattr(self,k,v)
    
    def __init__(self,config_data):
        """
        path_t_csv - путь до csv файла с текстами
        path_i_json - путь до json файла с картинками
        path_i_folder - путь для сохранения скаченных фотографий
        clastering_mode - тексты разбиты на кластеры
        """
        self.args = self.Args(config_data)
        
        self.text_data = pd.read_csv(self.args.path_t_csv)
        self.img_links = self._load_json_links(self.args.path_i_json) # links to download images
        
        
        if self.args.down_data:
            manager = mp.Manager()
            self._imgs_path = manager.Queue()
            self._load_imgs(list(self.img_links.items()),n_workers = 12)
            #self._check_data_in_folder()
            self._check_open_images(12) if self.args.check_img else True
            
        else:
            #check data
            self._check_open_images(12) if self.args.check_img else True
            self._check_data_in_folder()
            
        self._check_files = []
        # class count get for traning
        # for tokenizer text, depend on model
        self.tokenizer = get_tokenizer()
        self.transform = _transform(self.args.resize_img)
        
        
    def __len__(self,):
        return self.text_data.id_img.unique().shape[0]
    
    def _check_open_images(self,n_workers):
        for image_folder in self.args.path_i_folders:
            print(f"Start check images in {image_folder}")
            
            onlyfiles = [f"{image_folder}/{f}" for f in listdir(image_folder) if isfile(join(image_folder, f))]
            
            with mp.Pool(n_workers) as p:
                p.map(self._check_open_images_w, onlyfiles)
            
        print("Done") 
           
    def _check_open_images_w(self, i_file):
        path_i = f"{i_file}"
        img = cv2.imread(path_i, cv2.IMREAD_COLOR)
        
        if not hasattr(img, "shape"):
            print(f"del file {path_i}")
            os.remove(path_i)
            
                
    def __getitem__(self,idx):
        name_img = self.text_data.iloc[idx,1]
        path_img = self.text_data.iloc[idx,2]
        cls_id = self.text_data.iloc[idx,3]
        
        img = cv2.imread(f"{path_img}/{name_img}.jpg", cv2.IMREAD_COLOR)
        texts = self.text_data.iloc[idx][0].split("SEP")[1:] # get gt text
        
        assert hasattr(img,"shape"), print(path_img,name_img)
        #get one random text
        txt_idx = random.randint(0,len(texts)-1)
        text = texts[txt_idx]
        
        
        if self.args.mode == "Sber":
            input_ids, attention_mask = get_text_batch([text], self.tokenizer, self.args.len_seq)
            image = [Image.fromarray(img)] # get_image_batch take shape count_i,img_dat
            img_input = get_image_batch(image, self.transform)
                
        return (img_input, input_ids, attention_mask, torch.tensor([cls_id]))
        
        
    
    def _check_data_in_folder(self,top = 300):
        """
        Delete rows with csv file which no in folder.
        """
        #оставить в csv файле только те sample изображения которых есть в папке
        new_df = {col:[] for col in self.text_data.columns.values}
        all_files = {}
        df ={"id_img":[],
              "path":[]}
        
        for i, path in enumerate(self.args.path_i_folders):
            # из этого сделать df and merge with text_data to row id_img6 ant clean row then None
            for f in listdir(path):
                if isfile(join(path, f)):
                    df["id_img"].append(int(f[:-4]))
                    df["path"].append(path)
                    
        df_folder = pd.DataFrame.from_dict(df)
        
        finall_df = df_folder.set_index('id_img').join(self.text_data.set_index('id_img'),how='inner',rsuffix='_other')
        finall_df["id_img"] = finall_df.index
        
        finall_df = finall_df.set_index([pd.Index(range(len(finall_df)))])
        
        finall_df = finall_df.drop(labels = ["path_other"], axis = 1)
        
        finall_df = finall_df.reindex(columns=self.text_data.columns.values)
        
        finall_df = finall_df.drop_duplicates(subset = "id_img")
       
        self.text_data = finall_df
        
                
    
    def _load_json_links(self,data_path: str, only_i_from_csv = True)->Dict[int, Tuple[str,str]]:
        """
        load data with json to variable
        """
        data = {}
        only_csv_links = []
        with jsonlines.open(data_path) as reader:
            reader = tqdm(reader)
            for obj in reader:
                if obj['image'] not in data:
                    data[obj['image']] = obj['url']
                    
        if only_i_from_csv:
            #скачивать изображения принадлежащие csv
            only_csv_links = {idx: data[idx] for idx in self.text_data.id_img.unique()}
            return only_csv_links
        
        return data
    
    
    def _worker(self,task):
        paths_img = self._load_img(task)
        self._imgs_path.put(paths_img)
        
    
    def _load_img(self,links: Tuple[int,str])->int:
        try:
            response = requests.get(f"{links[1]}")
            with open(f"{self.args.path_to_load}/{links[0]}.jpg", "wb") as img:
                if response.content and response.status_code == 200:
                    img.write(response.content)
                    return links[0]
                else:
                    print(f"Oyy response empty, miss {links[0]}")
        except requests.exceptions.ConnectionError as e:
            print(f"Oyy, miss {links[0]}")
        return -1
            
    def _load_imgs(self, links:  List[Tuple[int,str]], n_workers = 1) -> bool:
        """
        download imgs from network
        """
        all_row = set(self.text_data.id_img.unique())
        return_row = set()
        all_len = len(all_row)
        with mp.Pool(n_workers) as p:
            p.map(self._worker, links)
            
        for _ in range(len(links)):
            return_row.add(self._imgs_path.get())
                
        all_row.difference_update(return_row)
        
        for row in all_row:
             self.text_data = self.text_data.drop(self.text_data[self.text_data.id_img == row].index)
        
        assert all_len - len(all_row) == len(self.text_data.id_img.unique())
        
        print(f"Download photo {all_len - len(all_row)} with {all_len} finish")
        return True


# In[ ]: