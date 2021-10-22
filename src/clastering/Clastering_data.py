#!/usr/bin/env python
# coding: utf-8

# In[80]:


#Кластеризация Piplines
#1 
# train BOF[:topN] 20k -> SentenceTransformer -> AgglomerativeClustering -> KNeighborsClassifier
# inference predict KNeighborsClassifier
#2 bs 700k sample -> SentenceTransformer -> BKmeans
#3 bs 50k sapmle -> SentenceTransformer -> AgglomerativeClustering




import sys
from itertools import chain
from typing import Dict
sys.path.append("/data/hdd1/brain/BraTS19/YandexCup/sentence-transformers")
sys.path.append("/data/hdd1/brain/BraTS19/YandexCup/src/utils")

import pandas as pd
import pickle


import torch 
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
import numpy as np

from Preprocessing_text import PreprocText,config_prep_text
from sentence_transformers import SentenceTransformer

config = {"path_model": "model/KnnClassif.pkl",
 "name_embeder": "all-MiniLM-L6-v2",
 "path_to_data": "Test.csv",
 "preproc_data": False,
}



class ClasteringData:
    
    def __init__(self,config:Dict,):
        class Args:
            def __init__(self,cfg):
                for name,value in cfg.items():
                    setattr(self,name,value)
                    
        self.args = Args(config)

        self.embedder = SentenceTransformer(self.args.name_embeder)
        
        with open(self.args.path_model, "rb") as m:
            self.clust_model = pickle.load(m)
            
        if self.args.preproc_data:
            config_prep_text["path_to_csv"]= config_prep_text["path_from_csv"] = self.args.path_to_data
            PreprocText(config_prep_text).processing_from_csv()
            
        self.data = pd.read_csv(self.args.path_to_data)
    
    def _chunk_data(self,chunk):
        df = self.data[chunk[0]:chunk[1]]
        texts = list(chain.from_iterable([text.split("SEP") for text in df["text"]]))
        class_count = [len(text.split("SEP")) for text in df["text"]]
        return texts, class_count
    
    def _mean_tensor(self,embedings,class_count):
        step = [0,0]
        new_emb = torch.zeros((len(class_count),embedings.shape[1]),)
        for i, cl_c in enumerate(class_count):
            if cl_c == 0:
                continue
            step[1] += cl_c
            new_emb[i] = torch.mean(embedings[step[0]:step[1]], 0)
            step[0] += cl_c
        return new_emb
    
    def cluster_data_pipline1(self,bs = 200000):
        chunk = [0, 0]
        cluster_assignment = []
        for i in range(1,len(self.data)//bs+2):
            chunk[1]+=bs
            data, class_count = self._chunk_data(chunk)
            # Corpus with example sentences
            corpus_embeddings = self.embedder.encode(data)
            # Normalize the embeddings to unit length
            corpus_embeddings = corpus_embeddings /  np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
            #means tesor
            corpus_embeddings = self._mean_tensor(torch.tensor(corpus_embeddings), class_count)
            # Perform kmean clustering
            cluster_assignment.append(self.clust_model.predict(corpus_embeddings))
            chunk[0] = chunk[1]
        cluster_assignment = list(chain.from_iterable(cluster_assignment))
        assert(len(self.data)) == len(cluster_assignment)
        self.data["id_claster"] = cluster_assignment


# In[84]:





