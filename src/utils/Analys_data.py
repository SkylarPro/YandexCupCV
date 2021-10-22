#!/usr/bin/env python
# coding: utf-8

# In[3]:


from plotly.offline import iplot
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import cufflinks


from typing import List,Tuple
from itertools import chain
import jsonlines
from tqdm import tqdm

import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer

cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')

def enable_plotly_in_cell():
    import IPython
    from plotly.offline import init_notebook_mode
    display(IPython.core.display.HTML('''<script src="/static/components/requirejs/require.js"></script>'''))
    init_notebook_mode(connected=False)


class Analys:
    
    def __init__(self, data_path: str = None):
        """
        data_path: str
            Path to json file with image and queries attr
        """
        self.data = self._load_data(data_path) if data_path else None
    
    def __len__(self,):
        return len(self.data)
    
    def plot_len_sent(self,corpus:List[str],
                      len_type:str = "words",
                      name_plot:str = "",
                      color_plot:str = "red"):
        
        """ 
        visualization of statistics on lenght sentense 
        """
        
        layout = go.Layout(
            title = f"Length{len_type} of the text"
        )
        dt = go.Box(
                y=self._len_sent_words(corpus) \
            if len_type == "words" else self._len_sent_char(corpus),
                name = 'Length of the text',
                marker = dict(
                    color = color_plot,
                )
            )
        fig = go.Figure(data=dt,layout=layout)
        iplot(fig, filename = f"Length{len_type} of the text")
    
    def plot_top_ngramm(self,corpus, ngrams:Tuple[int, int], 
                        n:int,
                        name_plot:str = "Ngramms",
                        
                       ):
        """ 
        visualization of statistics on top ngramm from sentense 
        """
        frequency_word = self.get_top_ngramm(corpus,ngrams,n)
        enable_plotly_in_cell()
        layout = go.Layout(
            title = f"{name_plot} of the text"
        )
        frequency_word = pd.DataFrame(frequency_word, columns = ['Text' , 'count'])
        
        frequency_word.groupby('Text').sum()['count'].sort_values(ascending=True).iplot(
            kind='bar', yTitle='Count', linecolor='black',
            color="blue", title=f"{name_plot}", orientation='h')
        return frequency_word

    
    def get_top_ngramm(self,corpus, ngrams = (1,1),n = None) -> List[Tuple[str, int]]:
        
        corpus = list(["SEP".join(text) for text in corpus]) if type(corpus[-1]) == list else corpus
        
        vec = CountVectorizer(ngram_range=ngrams).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq[:n]
    
    def data_to_csv(self,data:List[List[str]],ind:List[int], path_to_file:str):
        text_q = ['SEP'.join(text) for text in data]
        assert len(ind) == len(text_q)
        pd.DataFrame({"id_imgs":ind, 
                       "text":text_q}).to_csv(path_to_file,index = False)
        return True
        
    def get_random_data(self,data_size: int)->Tuple[List[List[str]], int]:
        ind = self._get_rand_idexs(data_size)
        return [self.data[i] for i in ind], ind
        
    def _get_rand_idexs(self,data_size: int)->List[int]:
        indeces = list(self.data.keys())
        np.random.shuffle(indeces)
        return indeces[:data_size]
        
    def _len_sent_char(self,data: List[str])-> List[int]:
        return list(chain.from_iterable([list(map(len,sent)) for sent in data]))
    
    def _len_sent_words(self,data: List[str])-> List[int]:
        return list(chain.from_iterable([list(map(len,list(map(str.split,sent)))) for sent in data]))
        
    def _load_data(self,data_path) -> List[str]:
        data = {}
        with jsonlines.open(data_path) as reader:
            if True:
                reader = tqdm(reader)
            for obj in reader:
                if obj['image'] not in data:
                    data[obj['image']] = obj['queries']
                
        return data

