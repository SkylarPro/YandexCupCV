#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append("/data/hdd1/brain/BraTS19/YandexCup")
import torch
from clip.model import VisualEncoder, TextEncoder, CLIP
from clip.origin.clip import load
from transformers import GPT2Model, GPT2Config,GPT2Tokenizer


def get_tokenizer(cache_dir, pretrained_model_name="sberbank-ai/rugpt3small_based_on_gpt2"):
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name, cache_dir = cache_dir)
    add_tokens = tokenizer.add_special_tokens({"bos_token": "<s>"})

    assert add_tokens == 0
    # add_tokens = tokenizer.add_special_tokens({"cls_token": "<case>"})
    add_tokens = tokenizer.add_special_tokens({"eos_token": "</s>"})
    assert add_tokens == 0
    add_tokens = tokenizer.add_special_tokens({"pad_token": "<pad>"})
    assert add_tokens == 0
    return tokenizer


def load_model(path_to_model = "./cache/models/allmodel.pt", path_to_tokenizer = "./cache/tokenizer"):
    
    visual_model, img_transform = load(path_to_model,jit = False)
    

    configuration = GPT2Config()
    configuration.vocab_size = 50264
    configuration.n_ctx = 2048
    configuration.n_positions = 2048
    text_model = GPT2Model(configuration)
    text_model.h = text_model.h[:8]
    
    
    visual_encoder = VisualEncoder(
        model=visual_model.visual,
        d_in=512,
        d_out=1024
    )
    text_encoder = TextEncoder(
        model=text_model,
        eos_token_id=2,
        d_in=768,
        d_out=1024
    )
    model = CLIP(
        visual_encoder=visual_encoder,
        text_encoder=text_encoder,
        img_transform=img_transform
    )
    sd = torch.load(path_to_model)
    model.load_state_dict(sd)
    
    tokenizer = get_tokenizer(path_to_tokenizer)
    
    return model, img_transform, tokenizer