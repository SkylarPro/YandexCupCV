#!/usr/bin/env python
# coding: utf-8

# In[2]:
import sys
import warnings
# sys.path.append("path/to/YandexCup")


from sentence_transformers import SentenceTransformer
from numpy import typing as npt
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from typing import List
import random

import numpy as np
from sklearn.manifold import TSNE


get_ipython().run_line_magic('matplotlib', 'inline')

class Visual_text_emb:
    
    def __init__(self,embeding:npt.ArrayLike = None, labels: npt.ArrayLike = None,
                 text:List[str] = None, squeeze_emb:npt.ArrayLike = None):
        """
        embeding: npt.ArrayLike
            vector representation of a word or sentence
        labels: npt.ArrayLike
            labels for draw different color class
        text: List[str]
            Use to annotate a plot
        """
        self.embeding = embeding.reshape(embeding.shape[0], -1) if embeding else None
        self.labels = labels
        self.texts = text
        self.squeeze_emb = squeeze_emb
        if embeding:
            self._set_squeeze_features()
        
    def _mean_texts(self,texts:List[str],class_count:List[int])->List[str]:
        """
        Called when class averaging is enabled, takes a random estimate from the class
        """
        step = [0,0]
        new_texts = []
        for i in class_count:
            step[1]+=i
            rand_text_cls = random.randint(step[0], step[1]-1)
            new_texts.append(texts[rand_text_cls])
            step[0] = step[1]
        return new_texts 
        
    def _mean_tensor(self,embedings, class_count:List[int]):
        """
        average the sensor by classes to save visualization space
        """
        step = [0,0]
        new_emb = np.zeros((len(class_count),embedings.shape[1]),)
        for i, cl_c in enumerate(class_count):
            if cl_c == 0:
                continue
            step[1] += cl_c
            new_emb[i] = np.mean(embedings[step[0]:step[1]], 0)
            step[0] += cl_c
        
        return new_emb

    def set_labels(self,labels):
        self.labels = labels
        
    def set_text_to_emb(self,texts:List[str], class_count:List[int]= None)->npt.ArrayLike:
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        embiddings = embedder.encode(texts)
        if not class_count:
            warnings.warn("push count class and in function if you want mean tensor between class")
            self.embeding = embiddings
        else:
            self.embeding = self._mean_tensor(embiddings,class_count)
            self.texts = self._mean_texts(texts,class_count)
            
        self._set_squeeze_features()
        
    def _set_squeeze_features(self,method = "TSNE",perplexity = 15,n_components = 2,init = "pca",n_iter = 3500):
        
        """ 
        shrinks the space to two features with TSNE algorithm for visualization
        """
        n = self.embeding.shape[0]
        if method == "TSNE": 
            tsne_model_en_2d = TSNE(perplexity=perplexity, n_components = n_components,
                                    init=init, n_iter=n_iter, random_state=32)
            
        self.squeeze_emb = tsne_model_en_2d.fit_transform(self.embeding.reshape(n,-1))
        
    def plot_embedings(self,title="Plot embedings",a = 0.8,size_annotate = 6,filename = None):
                       
        assert self.squeeze_emb.shape[1] < 3, f"squeeze_emb size should be [bs,2] not \
        [{self.squeeze_emb.shape[0]}, {self.squeeze_emb.shape[1]}]"
            
        colors = cm.rainbow(np.linspace(0, 1, max(self.labels)+1)) if self.labels \
        else cm.rainbow(np.linspace(0, 1, 2))
        for idx, embeddings in enumerate(self.squeeze_emb):
            x = embeddings[0]
            y = embeddings[1]
            plt.scatter(x, y, c=colors[self.labels[idx]] if self.labels else colors[0], alpha=a)
            if self.labels:
                plt.annotate(self.texts[idx]
                             , alpha=0.9, xy=(x, y), xytext=(2, 2),
                                 textcoords='offset points', ha='right', va='bottom', size=size_annotate)
        plt.legend(loc=4)
        plt.title(title)
        plt.grid(True)
        if filename:
            plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
        plt.show()