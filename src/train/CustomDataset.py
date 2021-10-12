#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import cv2
import numpy as np

import pandas as pd
from typing import List, Tuple, Dict


import multiprocessing  as mp
import requests
import jsonlines
from tqdm import tqdm
import time
import random
from PIL import Image


from os import listdir
import os

from os.path import isfile, join

import sys
sys.path.append("/data/hdd1/brain/BraTS19/YandexCup/ru-clip")

from clip.evaluate.utils import (
    get_text_batch, get_image_batch, get_tokenizer,
)



class Sent2textDataset(Dataset):
    
    def __init__(self,path_t_csv, path_i_json, 
                 path_i_folder, down_data = False,
                 n_classes = 20, args = None,
                 tokenizer = None, clastering_mode = False,
                 transform = None, mode = "Sber",
                ):
        """
        path_t_csv - путь до csv файла с текстами
        path_i_json - путь до json файла с картинками
        path_i_folder - путь для сохранения скаченных фотографий
        clastering_mode - тексты разбиты на кластеры
        """
        self.text_data = pd.read_csv(path_t_csv) if type(path_t_csv) == str else path_t_csv 
        self.img_links = self._load_json_links(path_i_json) # links to download images
        
        
        self.path_to_img = path_i_folder if path_i_folder else path_i_json
        
        if down_data:
            manager = mp.Manager()
            self._imgs_path = manager.Queue()
            self._load_imgs(list(self.img_links.items()),n_workers = 16)
            self._check_data_in_folder()
            self._check_open_images()
            
        else:
            #check data
            self._check_open_images()
            self._check_data_in_folder()
            
        self.clastering_mode = clastering_mode
        self.transform = transform
        self.args = args
        self.mode = mode
        # class count get for traning
        self.n_classes = n_classes
        # for tokenizer text, depend on model
        self.tokenizer = get_tokenizer() if tokenizer == None else tokenizer
        
        
    def __len__(self,):
        return self.text_data.id_img.unique().shape[0]
    
    def _check_open_images(self,):
        onlyfiles = [f for f in listdir(self.path_to_img) if isfile(join(self.path_to_img, f))]
        count = 0 
        for i_file in onlyfiles:
            path_i = f"{self.path_to_img}/{i_file}"
            img = cv2.imread(path_i, cv2.IMREAD_COLOR)
            
            if not hasattr(img,"shape"):
                os.remove(path_i)
                count+=1
        print(f"Broken {count} images")
        return True
    
    
    def _stack_texts(self,now_idx):
        """
        return text.shape(1, self.n)
        """
        # get text with differend class or with currently class
        if self.clastering_mode:
            pass
        else:
            indexs = [random.randint(0,len(self.text_data)-1) for i in range(self.n_classes-1)]
            #проверка на совпадение индексов
            for i in range(len(indexs)):
                if indexs[i] == now_idx:
                    indexs[i] = now_idx + 1
            
            texts = []
            for i in indexs:
                t = self.text_data.iloc[i][0].split("SEP")[1:]
                t = t[random.randint(0,len(t)-1)]
                texts.append(t)
                
            return texts 
    
    
    def __getitem__(self,idx):
        name_img = self.text_data.iloc[idx,1]
                
        img = cv2.imread(f"{self.path_to_img}/{name_img}.jpg", cv2.IMREAD_COLOR)
        text = self.text_data.iloc[idx][0].split("SEP")[1:] # get gt text
        
        #get one random text
        txt_idx = random.randint(0,len(text)-1)
        text = text[txt_idx]
        
        texts = self._stack_texts(idx) # create new class
        
        gt_idx = random.randint(0,self.n_classes - 1)
        texts.insert(gt_idx, text)
        
        
        if self.mode == "Sber":
            assert self.args != None, f"Define args"
            input_ids, attention_mask = get_text_batch([text], self.tokenizer, self.args)
            if self.transform == None:
                image = [Image.fromarray(img)] # get_image_batch take shape count_i,img_dat
                img_input = get_image_batch(image, self.args.img_transform, self.args)
            else:
                img_input = self.transform(img)
                
        return (img_input, input_ids, attention_mask) #, gt_idx, text
        
        
    
    def _check_data_in_folder(self,):
        """
        Delete rows with csv file which no in folder.
        """
        #оставить в csv файле только те sample изображения которых есть в папке
        onlyfiles = {int(f[:-4]): True for f in listdir(self.path_to_img) if isfile(join(self.path_to_img, f))}
        
        new_df = pd.DataFrame([], columns = self.text_data.columns.values)
        data = []
        for id_img in onlyfiles:
            row = self.text_data[self.text_data["id_img"] == id_img]
            if row.shape[0] != 0:
                data.append(row)
                
        new_df = new_df.append(data, ignore_index=True)
                
        print(f"{new_df.shape[0]} images in folder from {self.text_data.shape[0]} in csv file")
        self.text_data = new_df
        
                
    
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
            with open(f"{self.path_to_img}/{links[0]}.jpg", "wb") as img:
                if response.content and response.status_code == 200:
                    img.write(response.content)
                    return links[0]
                else:
                    print(f"Oyy response empty, miss {links[0]}")
        except requests.exceptions.ConnectionError as e:
            print(f"Oyy, miss {links[0]}")
            
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