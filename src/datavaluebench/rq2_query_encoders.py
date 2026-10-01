"""Reusable DataValueBench benchmark implementation."""
from pathlib import Path
import numpy as np,torch

class E5QueryEncoder:
    def __init__(self,snapshot):
        from transformers import AutoTokenizer,AutoModel
        self.tokenizer=AutoTokenizer.from_pretrained(Path(snapshot),local_files_only=True)
        self.model=AutoModel.from_pretrained(Path(snapshot),local_files_only=True,torch_dtype=torch.float32).eval()
        if self.tokenizer.truncation_side!='right':raise ValueError('Frozen right truncation required')
    def encode(self,texts):
        if not texts or any(not isinstance(x,str) for x in texts):raise TypeError('Exact NL strings only')
        tokens=self.tokenizer(['query: '+x for x in texts],max_length=512,padding=True,truncation=True,return_tensors='pt')
        with torch.inference_mode():
            hidden=self.model(**tokens).last_hidden_state
            hidden=hidden.masked_fill(~tokens['attention_mask'][...,None].bool(),0.)
            pooled=hidden.sum(1)/tokens['attention_mask'].sum(1)[...,None]
            vectors=torch.nn.functional.normalize(pooled,p=2,dim=1)
        return vectors.cpu().numpy()

class SPLADEQueryEncoder:
    def __init__(self,snapshot):
        from sentence_transformers import SparseEncoder
        self.model=SparseEncoder(str(snapshot),device='cpu',local_files_only=True)
        self.model.max_seq_length=256;self.model.eval();self.tokenizer=self.model.tokenizer
        if self.model[1].pooling_strategy!='max' or self.model[1].activation_function!='relu':raise ValueError('Frozen max/ReLU pooling required')
        if self.tokenizer.truncation_side!='right':raise ValueError('Frozen right truncation required')
    def encode(self,texts):
        if not texts or any(not isinstance(x,str) for x in texts):raise TypeError('Exact NL strings only')
        with torch.inference_mode():
            return self.model.encode_query(texts,batch_size=8,show_progress_bar=False,convert_to_tensor=True,convert_to_sparse_tensor=False,max_active_dims=None).cpu().numpy()
