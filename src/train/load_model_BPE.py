#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import torch
# sys.path.append("path/to/YandexCup")
from clip.model import VisualEncoder, TextEncoder, CLIP
from clip.origin.clip import load
from transformers import GPT2Model, GPT2Config,GPT2Tokenizer
from bpemb import BPEmb


def get_tokenizer(path_to_token = "cache/tokenizer/BPE/ru.wiki.bpe.vs200000.model",):
    tokenizer = BPEmb(lang="ru", dim=200, vs = 200000,segmentation_only = True, model_file = path_to_token)
    return tokenizer


def load_model(path_to_model = "./cache/models/GPTsmall_epoch_1id4_ds_N2_from_BPEV2.pt", path_to_tokenizer = "./cache/tokenizer"):
    
    visual_model, img_transform = load(path_to_model,jit = False)
    

    configuration = GPT2Config()
    configuration.vocab_size = 200000
    configuration.n_embd = 200 
    configuration.n_head = 10
    configuration.n_ctx = 2048
    configuration.n_positions = 2048
    text_model = GPT2Model(configuration)
    
    
    visual_encoder = VisualEncoder(
        model=visual_model.visual,
        d_in=512,
        d_out=512
    )
    text_encoder = TextEncoder(
        model=text_model,
        eos_token_id=2,
        d_in=200,
        d_out=512
    )
    model = CLIP(
        visual_encoder=visual_encoder,
        text_encoder=text_encoder,
        img_transform=img_transform
    )
    sd = torch.load(path_to_model, map_location=torch.device('cpu'))
    model.load_state_dict(sd)
    
    tokenizer = get_tokenizer()
    
    return model, img_transform, tokenizer
