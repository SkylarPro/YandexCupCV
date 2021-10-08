#!/usr/bin/env python
# coding: utf-8

# In[2]:


from numpy import typing as npt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
get_ipython().run_line_magic('matplotlib', 'inline')

class VisualEmb:
    
    def __init__(self,embeding:npt.ArrayLike, labels: npt.ArrayLike,
                text:List[str] = None, squeeze_emb:npt.ArrayLike = None):
        
        self.labels = labels
        self.embeding = embeding
        self.text = text
        self.squeeze_emb = squeeze_emb
        
    def _squeeze_features(self,method = "TSNE",perplexity = 15,n_components = 2,init = "pca",n_iter = 3500):
        n = self.embeding.shape[0]
        if method == "TSNE": 
            tsne_model_en_2d = TSNE(perplexity=perplexity, n_components = n_components,
                                    init=init, n_iter=n_iter, random_state=32)
            
            self.squeeze_emb = tsne_model_en_2d.fit_transform(self.embeding.reshape(n,-1))
    
    @property
    def plot_embedings(self,title="Plot embedings",a = 0.5,filename = None):
        
        if not isinstance(type(self.squeeze_emb), type(np.ndarray)):
            self._squeeze_features()
            
        colors = cm.rainbow(np.linspace(0, 1, max(self.labels)+1))
        
        for idx, embeddings in enumerate(self.squeeze_emb):
            x = embeddings[0]
            y = embeddings[1]
            plt.scatter(x, y, c=colors[self.labels[idx]], alpha=a)
            plt.annotate(self.labels[idx]
                         , alpha=0.9, xy=(x, y), xytext=(2, 2),
                             textcoords='offset points', ha='right', va='bottom', size=6)
        plt.legend(loc=4)
        plt.title(title)
        plt.grid(True)
        if filename:
            plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
        plt.show()
