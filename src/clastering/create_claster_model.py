#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import numpy as np


from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
import pickle

import sys
# sys.path.append("path/to/YandexCup")
from sentence_transformers import SentenceTransformer

config_clater_model = {
    "path_to_data":"data/BOW2_3mln.csv",
    "name_embeder": "all-MiniLM-L6-v2",
    "save_example":"Example.csv",
    "n_clasters":10,
    "n_neighbors":4,
    "path_to_model":"model/Testcls.pkl"
}

class Claster_model:
    
    def __init__(self,config):
        
        class Args:
            def __init__(self,cfg):
                for name,val in cfg.items():
                    setattr(self,name,val)
                    
        self.args = Args(config)
        
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.data = pd.read_csv(self.args.path_to_data)
    
    def _save_prdict_train(self,cluster_assignment):
        """
        Save clastering sentence on train data
        """
        df = pd.read_csv(self.args.path_to_data)
        assert len(cluster_assignment) == len(df)
        df["id_claster"] = cluster_assignment
        df.to_csv(self.args.path_to_data, index = False)
        return True
    
        
        
    def create_model(self,):
        texts = self.data.Text.to_numpy()
        corpus_embeddings = self.embedder.encode(texts)
        corpus_embeddings = corpus_embeddings /  np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
        
        #us
        clustering_model = AgglomerativeClustering(n_clusters=self.args.n_clasters,)
        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_
        self._save_prdict_train(cluster_assignment)
        assert len(cluster_assignment) == len(corpus_embeddings)
        
        X_train, X_test, y_train, y_test = train_test_split(corpus_embeddings,cluster_assignment,shuffle=True,
                                                            test_size=0.1,random_state=42)
        
        model = KNeighborsClassifier(n_neighbors = self.args.n_neighbors)
        model.fit(X_train,y_train)
        
        pr = model.predict(X_test)
        print(accuracy_score(pr,y_test))
        
        pickle.dump(model,open(self.args.path_to_model,"wb"))


# In[ ]:




