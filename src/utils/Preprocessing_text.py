from itertools import chain
import pandas as pd
#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pymorphy2
import multiprocessing as mp
from nltk.tokenize import TweetTokenizer
from typing import Dict, List,Tuple
import re
import string


config_prep_text = {
    "path_to_csv":"Output.csv",
    "path_from_csv":"Input.csv",
    "remove_input_data":False,
    "lower_text":True,
    "only_ru_simb":True,
    "clean_space": True,
    "clean_link": True,
    "clean_hashtag": True,
    "clean_punct": True,
    "word_to_lemma": True,
    "min_len_sent" :1,
    "stopwrd": [],
}

class PreprocText:
    
    def __init__(self,config:Dict[str, bool], tokenizer=None,):
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = TweetTokenizer(preserve_case=False,strip_handles=True,
                                                                    reduce_len=True,                                    
                                                                   )
        self.config = config
        
        m = mp.Manager()
        self._result_queue = m.Queue()
        self.procc_text = []
        
        self.balance_class = []
        self._morph = pymorphy2.MorphAnalyzer(lang='ru')
        self.empty_class = []
        
    def _processing_text(self,proc_text):
        idx = proc_text[0]
        proc_text = proc_text[1]
        proc_text = proc_text.lower() if self.config.get("lower_text") == True else proc_text
        proc_text = re.sub('[^а-я]', " ",proc_text) if self.config.get("only_ru_simb") == True else proc_text
        proc_text = proc_text.replace("  ", " ") if self.config.get("clean_space") == True else proc_text
        
        proc_text = re.sub(r"http\S+", "",proc_text) if self.config.get("clean_link") == True else proc_text
        proc_text = re.sub(r'#','',proc_text) if self.config.get("clean_hashtag") == True else proc_text
    
        proc_text = [char for char in proc_text if char not in string.punctuation] if self.config.get("clean_punct") == True else proc_text
        proc_text = ''.join(proc_text)
        
        
        proc_text =  ' '.join([word for word in proc_text.split() if word.lower() not in self._stpword]) if len(self.config.get("stopwrd")) != 0  else proc_text
        
        if self.config.get("word_to_lemma") == True:
            proc_text = self.tokenizer.tokenize(proc_text)
            sent_lemm = []
        
            for word in proc_text:
                word_normal = self._morph.parse(word)[0]
                #confidenc model to cast form word
                if word_normal.score > 0.70:
                    sent_lemm.append(word_normal.normal_form)
                else:
                    sent_lemm.append(word)
            proc_text = ' '.join(sent_lemm)
                
        return idx, proc_text
    
    def _worker(self, task):
        result = self._processing_text(task)
        self._result_queue.put(result)
    
    
    def _balans_start_index(self,class_count):
        """
           Some sentences became empty after processing, 
           they need to be removed from the classes and
           as a consequence, change the number of elements in the class.
        """
        
        data_proc = []
        step = [-1, 0]
        min_len_sent = self.config.get("min_len_sent")
        
        for idx, (_,text) in enumerate(self.procc_text):
            
            if step[1] == idx:
                step[0] += 1
                step[1] += class_count[step[0]]

            if len(text.split()) <= min_len_sent:
                # минимальное количество слов в предложении
                class_count[step[0]] -= 1
            else:
                data_proc.append(text)
                
            if class_count[step[0]] == 0:
                self.empty_class.append(step[0])
                
        assert len(data_proc) == sum(class_count)
        
        return data_proc, class_count
        
    
    @property
    def get_data(self,):
        return self.procc_text, self.balance_class
    
    def save_in_csv(self,id_img,remove_input_file = False):
        idxs = list(chain.from_iterable([[label] * count for label, count in zip(id_img, self.balance_class)]))
        data = {}
        for idx, text in zip(idxs, self.procc_text):
            if idx not in data:
                data[idx] = []
            data[idx].append(text)
        data = {key:"SEP".join(data[key]) for key,val in data.items()}
        pd.DataFrame({"id_imgs":data.keys(),
                      "text": data.values()
                     }).to_csv(self.config["path_to_csv"],index = False)
        if self.config["remove_input_data"]:
            os.remove(self.config["path_from_csv"])
        return True
    
    
    def processing_from_csv(self,):
        df = pd.read_csv(self.config["path_from_csv"])
        texts = [text.split("SEP") for text in df["text"]]
        class_count = [len(sample) for sample in texts]
        
        data = list(zip(range(sum(class_count)),chain.from_iterable(texts)))
        id_imgs = df["id_imgs"].values
        assert len(data) == sum(class_count),print(len(data), sum(class_count))
        self.processing_big_data(data,class_count = class_count,id_img = id_imgs)
        
    
    def processing_big_data(self,data, class_count = None, id_img = None, n_worker = 1):
        
        with mp.Pool(n_worker) as p:
            p.map(self._worker, data)
        
        for _ in range(len(data)):
            self.procc_text.append(self._result_queue.get())
            
        assert len(self.procc_text) == len(data), f"{len(self.procc_text)} != {len(data)}"
        print("Started sorting")
        self.procc_text = sorted(self.procc_text)
        if class_count:
            self.procc_text, self.balance_class = self._balans_start_index(class_count)
        else: 
            self.procc_text, self.balance_class = self.procc_text, None
        if self.config["path_to_csv"].find(".csv")!=-1:
            self.save_in_csv(id_img)
        return self.procc_text, self.balance_class