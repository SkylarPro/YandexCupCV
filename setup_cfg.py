#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from clip.evaluate.utils import (
    load_weights_only,
)
from clip.evaluate.utils import (
    get_tokenizer,
)

if __name__ == "__main__":
    model, args = load_weights_only("ViT-B/32-small",seq_length = 15)
    get_tokenizer()


# In[1]:


jupyter nbconvert --to python setup_cfg.ipynb


# In[ ]:




