"""Reusable DataValueBench benchmark implementation."""
from itertools import combinations
import numpy as np
FIELDS=('task_categories','task_contexts','tags','language','license')
class HFStructuredDiversity:
    def __init__(self,records):
        self.values={}
        for row in records:
            identity=row['dataset_id']
            if not isinstance(identity,str) or not identity or identity in self.values:raise ValueError('Invalid/duplicate canonical HF identity')
            fields=[]
            for name in FIELDS:
                value=row[name+'_values']
                if not isinstance(value,(list,tuple,np.ndarray)) or any(not isinstance(x,str) for x in value):raise ValueError('Canonical set-valued metadata required')
                fields.append(frozenset(value))
            self.values[identity]=tuple(fields)
    def distance(self,left,right):
        a=self.values[left];b=self.values[right]
        distances=[0. if not x and not y else 1.-len(x&y)/len(x|y) for x,y in zip(a,b)]
        return float(np.mean(distances,dtype=np.float64))
    def ild(self,candidates):
        if len(candidates)<2 or len(set(candidates))!=len(candidates):raise ValueError('At least two distinct candidates required')
        return float(np.mean([self.distance(x,y) for x,y in combinations(candidates,2)],dtype=np.float64))
