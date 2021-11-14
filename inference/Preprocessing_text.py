#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from itertools import chain
import pandas as pd
#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pymorphy2
import multiprocessing as mp
from nltk.tokenize import TweetTokenizer
from typing import Dict, List
import re
import string


config_prep_text = {
    "lower_text":True,
    "only_ru_simb":True,
    "clean_space": True,
    "clean_link": True,
    "clean_hashtag": True,
    "clean_punct": True,
    "clean_stw": True,
    "word_to_lemma": False,
    "min_len_sent" :1,
}

class PreprocText:
    
    def __init__(self,config:Dict[str, bool], tokenizer=None,stpword: List[str] = []):
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
        self._stpword = stpword
        self.empty_class = []
        
    def _processing_text(self,proc_text):
        proc_text = proc_text.lower() if self.config.get("lower_text") == True else proc_text
        proc_text = re.sub('[^а-я]', " ",proc_text) if self.config.get("only_ru_simb") == True else proc_text
        proc_text = proc_text.replace("  ", " ") if self.config.get("clean_space") == True else proc_text
        
        proc_text = re.sub(r"http\S+", "",proc_text) if self.config.get("clean_link") == True else proc_text
        proc_text = re.sub(r'#','',proc_text) if self.config.get("clean_hashtag") == True else proc_text
    
        proc_text = [char for char in proc_text if char not in string.punctuation] if self.config.get("clean_punct") == True else proc_text
        proc_text = ''.join(proc_text)
        
        proc_text =  ' '.join([word for word in proc_text.split() if word.lower() not in self._stpword]) if self.config.get("clean_stw") == True  else proc_text
        
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
                
        return proc_text
    
    def _worker(self, task):
        result = self._processing_text(task)
        self._result_queue.put(result)
    
    
    def _balans_start_index(self,class_count):
        """
        Некоторые предложения стали пустыми после обработки,
        их надо убрать из классов и 
        как следствие изменить количество элементво в классе.
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
    
    def processing_big_data(self,data, class_count = None, n_worker = 1):
        
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
        return self.procc_text, self.balance_class


# In[1]:





