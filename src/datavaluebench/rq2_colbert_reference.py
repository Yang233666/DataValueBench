"""Reusable DataValueBench benchmark implementation."""
import ast,string,types
from pathlib import Path
import torch,transformers
import torch.nn as nn
from transformers import AutoConfig,AutoTokenizer
from safetensors.torch import load_file

def _load_nodes(path,names,namespace,*,class_methods=None):
 tree=ast.parse(Path(path).read_text());nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
 if class_methods:
  cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==class_methods[0]);nodes += [n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in class_methods[1]]
 if not nodes:raise ValueError('required official reference definitions missing')
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
 return namespace

class FrozenColBERT:
 def __init__(self,snapshot,official_source):
  snapshot=Path(snapshot);official_source=Path(official_source);config=AutoConfig.from_pretrained(snapshot,local_files_only=True)
  if config.model_type!='bert' or getattr(config,'auto_map',None) is not None:raise ValueError('only frozen BERT ColBERT checkpoint is allowed')
  self.config=types.SimpleNamespace(checkpoint=str(snapshot),dim=128,query_maxlen=32,doc_maxlen=180,query_token='[Q]',doc_token='[D]',query_token_id='[unused0]',doc_token_id='[unused1]',attend_to_mask_tokens=False,mask_punctuation=True,interaction='colbert',similarity='cosine',total_visible_gpus=0)
  ns={'torch':torch,'nn':nn,'transformers':transformers,'transformers_module':dir(transformers),'AutoConfig':AutoConfig,'AutoTokenizer':AutoTokenizer,'ColBERTConfig':types.SimpleNamespace,'DEVICE':torch.device('cpu'),'base_class_mapping':{},'model_object_mapping':{}}
  _load_nodes(official_source/'modeling/hf_colbert.py',{'find_class_names','class_factory'},ns)
  factory=ns['class_factory'](str(snapshot));hf=factory(config,self.config)
  state=load_file(str(snapshot/'model.safetensors'))
  # Older Transformers serialized this fixed buffer; newer versions retain the
  # same buffer but omit it from state_dict. Preserve and strictly load it.
  name='bert.embeddings.position_ids';embedding=hf.bert.embeddings
  if name in state:
   if not torch.equal(state[name],embedding.position_ids):raise ValueError('checkpoint position_ids differs from exact runtime buffer')
   embedding._non_persistent_buffers_set.discard('position_ids')
  hf.load_state_dict(state,strict=True);hf.eval();self.hf=hf
  _load_nodes(official_source/'modeling/tokenization/utils.py',{'_insert_prefix_token','_split_into_batches','_sort_by_length'},ns)
  _load_nodes(official_source/'modeling/tokenization/doc_tokenization.py',{'DocTokenizer'},ns);_load_nodes(official_source/'modeling/tokenization/query_tokenization.py',{'QueryTokenizer'},ns)
  self.doc_tokenizer=ns['DocTokenizer'](self.config);self.query_tokenizer=ns['QueryTokenizer'](self.config,verbose=0);tok=self.doc_tokenizer.tok
  assert tok.truncation_side=='right' and self.doc_tokenizer.D_marker_token_id==2 and self.query_tokenizer.Q_marker_token_id==1
  _load_nodes(official_source/'modeling/colbert.py',{'colbert_score','colbert_score_reduce'},ns,class_methods=('ColBERT',{'query','doc','mask'}))
  adapter=types.SimpleNamespace(bert=hf.LM,linear=hf.linear,device=torch.device('cpu'),pad_token=tok.pad_token_id,use_gpu=False,colbert_config=self.config)
  adapter.skiplist={w:True for symbol in string.punctuation for w in [symbol,tok.encode(symbol,add_special_tokens=False)[0]]}
  adapter.mask=types.MethodType(ns['mask'],adapter);self._doc=types.MethodType(ns['doc'],adapter);self._query=types.MethodType(ns['query'],adapter);self._score=ns['colbert_score']
 def docs(self,texts):
  ids,mask=self.doc_tokenizer.tensorize(texts)
  with torch.inference_mode():vectors,valid=self._doc(ids,mask,keep_dims='return_mask')
  return vectors,valid,ids,mask
 def queries(self,texts):
  ids,mask=self.query_tokenizer.tensorize(texts)
  with torch.inference_mode():vectors=self._query(ids,mask)
  return vectors,ids,mask
 def score(self,query,docs,mask):return self._score(query,docs,mask,config=self.config)
