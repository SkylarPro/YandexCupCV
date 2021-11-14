import torch
from torch import nn
import numpy as np
from .origin.clip import load
from transformers import GPT2Model
from PIL import Image
from .origin.model import Transformer
from collections import OrderedDict


def gelu(x):
    return x * torch.sigmoid(1.702 * x)


class Projection(nn.Module):
    def __init__(self, d_in: int, d_out: int, p=0.5) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds


class VisualEncoder(nn.Module):
    def __init__(self, model, d_in, d_out):
        super().__init__()
        self.model = model
        self.projection = Projection(d_in, d_out)

    def forward(self, x):
        x = self.model(x)
        # print(x.shape)
        x = self.projection(x)
        projection_len = torch.norm(x, dim=-1, keepdim=True)
        return x / projection_len
    
    
class Identity(nn.Module):
    def __init__(self,):
        super().__init__()
        
    def forward(self, inp):
        # Remove the following line to get weird behavior
        inp = inp.clone()
        return inp
    
    
class TextEncoder(nn.Module):
    def __init__(self, model, eos_token_id, d_in, d_out, count_class = 10):
        super().__init__()
        self.model = model
        self.eos_token_id = eos_token_id
        self.count_class = count_class
        
        if count_class:
            self.distribution_cls = nn.Sequential(OrderedDict([
              ('input_cls', Transformer(768,1,12)),
              ('output_cls', nn.Linear(768, count_class)),
            ]))
            
            self.projection = self._make_layers_projection(d_in, d_out, count_class)
        else:
            self.projection = Projection(d_in, d_out)

        
    def _make_layers_projection(self,d_in,d_out,count_class):
        layers_proj = [Projection(d_in, d_out) if i%2 == 0  else Identity() for i in range(count_class*2)]
        return nn.Sequential(*layers_proj[:-1])
    
    
    def forward(self, text: torch.Tensor, **kwargs):
        
        
        x = self.model(text, **kwargs)[0][(text == self.eos_token_id).nonzero(as_tuple=True)]
        
        if self.count_class:
            embeding = self.model.wte(text)
            embeding = embeding.permute(1, 0, 2)
            x_cls = self.distribution_cls[0](embeding)
            x_cls = x_cls.permute(1, 0, 2)
            
            x_cls = x_cls[torch.arange(x_cls.shape[0]), text.argmax(dim=-1)]
            
            x_cls = self.distribution_cls[1](x_cls)
            
            _, cls = torch.max(x_cls, dim = 1)
            
            
            x_final = self.projection[cls[0]*2](x[0]).reshape(1,-1)
        
            for i, cl in enumerate(cls[1:],1):
                x_final = torch.cat((x_final,self.projection[cl*2](x[i]).reshape(1,-1).clone()),dim = 0)
                
            projection_len = torch.norm(x_final, dim=-1, keepdim=True)
            return (x_final / projection_len), x_cls
            
        projection_len = torch.norm(x, dim=-1, keepdim=True)
        return (x_final / projection_len), x_cls


class CLIP(nn.Module):
    def __init__(self, visual_encoder, text_encoder, img_transform):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.text_encoder = text_encoder
        self.img_transform = img_transform
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, img_input, text_input):
        image_features = self.visual_encoder(**img_input)
        text_features,cls = self.text_encoder(**text_input)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()
        return logits_per_image, logits_per_text, cls


def get_model(args):
    visual_model, args.img_transform = load(args.visual_encoder_name, jit=False)
    text_model = GPT2Model.from_pretrained(args.load_huggingface)

    visual_encoder = VisualEncoder(
        model=visual_model.visual,
        d_in=args.visual_encoder_dim,
        d_out=args.clip_projection_dim
    )
    text_encoder = TextEncoder(
        model=text_model,
        eos_token_id=args.eos_token_id,
        d_in=args.hidden_size,
        d_out=args.clip_projection_dim
    )
    model = CLIP(
        visual_encoder=visual_encoder,
        text_encoder=text_encoder,
        img_transform=args.img_transform
    )
    if args.freeze_visual_encoder:
        for p in model.visual_encoder.model.parameters():
            p.requires_grad = False

    if args.freeze_text_encoder:
        for p in model.text_encoder.model.parameters():
            p.requires_grad = False

    # To prevent OOM for model sizes that cannot fit in GPU memory in full precision
    if args.deepspeed and args.fp16:
        model.half()

    if not args.cpu:
        model.cuda(torch.cuda.current_device())

    return model


def get_image_batch(img_paths, img_transform, args = None):
    images = []
    for path in img_paths:
        if isinstance(path, Image.Image):
            image = path
        else:
            image = Image.open(path)
        image = image.convert("RGB")
        image = img_transform(path)
        images.append(image)
    images = torch.tensor(np.stack(images))
    return images
