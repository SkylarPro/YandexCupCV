#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
# sys.path.append("path_to_YandexCup")
from os.path import isfile 

import torch
from clip.model import VisualEncoder, TextEncoder, CLIP
from clip.origin.clip import load
from transformers import GPT2Model, GPT2Config,GPT2Tokenizer


def get_tokenizer(vocab = "./cache/tokenizer/GPT2_small/vocab.json", merges="./cache/tokenizer/GPT2_small/merges.txt"):
    tokenizer = GPT2Tokenizer(vocab, merges)
    add_tokens = tokenizer.add_special_tokens({"bos_token": "<s>"})

    assert add_tokens == 0
    # add_tokens = tokenizer.add_special_tokens({"cls_token": "<case>"})
    add_tokens = tokenizer.add_special_tokens({"eos_token": "</s>"})
    assert add_tokens == 0
    add_tokens = tokenizer.add_special_tokens({"pad_token": "<pad>"})
    assert add_tokens == 0
    return tokenizer


def load_model(path_to_model = None ,count_class = 10):
    print(os.getcwd())
    
    visual_model, img_transform = load(path_to_model,jit = False)
    

    configuration = GPT2Config()
    configuration.vocab_size = 50264
    configuration.n_ctx = 2048
    configuration.n_positions = 2048
    
    text_model = GPT2Model(configuration)
    
    
    visual_encoder = VisualEncoder(
        model=visual_model.visual,
        d_in=512,
        d_out=1024
    )
    text_encoder = TextEncoder(
        model=text_model,
        eos_token_id=2,
        d_in=768,
        d_out=1024,
        count_class = count_class,
    )
    model = CLIP(
        visual_encoder=visual_encoder,
        text_encoder=text_encoder,
        img_transform=img_transform
    )
    if isfile(path_to_model):
        sd = torch.load(path_to_model)
        model.load_state_dict(sd)
    else:
        print("Weight don't init")
        
    tokenizer = get_tokenizer()
    
    return model, img_transform, tokenizer