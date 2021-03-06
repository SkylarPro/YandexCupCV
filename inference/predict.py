# hydra imports
# from omegaconf import OmegaConf

# generic imports
from typing import Optional, List, Iterable
from tqdm.auto import tqdm
from PIL import Image
import numpy as np
import more_itertools
import os
import json
import click
import warnings
import sys

from clip.evaluate.utils import (
    get_text_batch, get_image_batch,
)

from Preprocessing_text import PreprocText


# torch imports
import torch
from torch.utils.data._utils.collate import default_collate
from load_model import load_model
# custom imports



class I2TInferer(object):
    def __init__(
        self,
        ckpt_path: str = None,
        device: str = None,
        ):
        
        #load tools
        print(ckpt_path)
        model, img_transfrom, text_tokenizer = load_model(ckpt_path)
        self.len_seq = 15

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
            "stopwrd":[],
        }

        self.proc_text = PreprocText(config_prep_text)
        self.tokenizer = text_tokenizer
        self.image_transform = img_transfrom
        
        model = model.eval()
        model = model.to(device)

        self.model = model
        self.logit_scale = model.logit_scale
        self.device = device
        
    def encode_texts(self, texts: Iterable[str]) -> torch.Tensor:
        texts = list(map(self.proc_text._processing_text,texts))
        input_ids, attention_mask = get_text_batch(texts, self.tokenizer, self.len_seq)
        
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        return self.model.text_encoder(**{"x": input_ids,"attention_mask":attention_mask})
    
    def encode_images(self, images: Iterable[Image.Image]) -> torch.Tensor:
        pbar = tqdm()
        image_features = []
        for chunk in more_itertools.chunked(images, 10):
            images = [self.image_transform(x.convert('RGB')) for x in chunk]
            
            images = torch.tensor(np.stack(images))
            images = images.to(self.device)
            
            chunk_image_features = self.model.visual_encoder(images).cpu().detach()
            image_features.append(chunk_image_features)
            pbar.update(len(chunk))
        pbar.close()
        return torch.cat(image_features, dim=0)
    
    def predict(self, images: Iterable[Image.Image], classes: Iterable[str]) -> np.ndarray:
        text_features = self.encode_texts(classes) # return text_features and cls 
        image_features = self.encode_images(images)
        
        logits_per_text = self.logit_scale * image_features @ text_features.t()
        
        _, preds = torch.max(logits_per_text, 1)
        
        return preds.cpu().numpy()


@click.command()
@click.option('--ckpt_path', help='Path to PL checkpoint')
@click.option('--data_directory', help='Path to directory with evaluation datasets')
@click.option('--predicts_file', help='Path to file where predictions should be put to')
@click.option('--limit_samples', default=None, type=int, help='Limit num of evaluated images')
@click.option('--device', default='cpu', help='PyTorch device')
@click.option('--num_threads', default=None, type=int, help='Optionally force number of torch threads')
@click.option('--dataset', '-d', default=None, multiple=True, help='Optionally select datasets manually')
@torch.no_grad()
def main(
    ckpt_path: str,
    data_directory: str,
    predicts_file: str,
    limit_samples: Optional[int],
    device: str,
    num_threads: Optional[int],
    dataset: Optional[List[str]]
):
    warnings.simplefilter('always', DeprecationWarning)
    if num_threads is not None:
        torch.set_num_threads(num_threads)
    
    
    inferer = I2TInferer(ckpt_path=ckpt_path, device=device)
    if dataset:
        datasets = dataset
    else:
        datasets = os.listdir(data_directory)

    results = {}
    for dataset in datasets:
        with open(f"{data_directory}/{dataset}/classes.json", 'r') as f:
            classes_labels = json.load(f)
        image_files = os.listdir(f'{data_directory}/{dataset}/img')
        if limit_samples is not None:
            image_files = image_files[:limit_samples]
            
        images = (Image.open(f'{data_directory}/{dataset}/img/{file}') for file in image_files)
        predicts = inferer.predict(images, classes_labels).tolist()
        warnings.warn(f"Count imgs {len(image_files)} count predicts {len(predicts)}", DeprecationWarning)
        
        predicts = {file.split('.')[0]: predict for file, predict in zip(image_files, predicts)}
        results[dataset] = predicts
    warnings.warn(f"{results}", DeprecationWarning)
    with open(predicts_file, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
