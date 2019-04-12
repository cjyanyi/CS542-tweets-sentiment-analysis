# -*- coding:UTF-8 -*-

from pred import TweetsAnalysis
from numpy import genfromtxt
import numpy as np

def read_crawler_csv(fname=''):
    return genfromtxt(fname, delimiter=',', skip_header=0)

def parse_preds(preds):
    neg_n = pos_n = neut_n = 0
    for pred in preds:
        if pred[0]>0.6:
            neg_n+=1
        elif pred[1]>0.6:
            pos_n+=1
        else:
            neut_n+=1
    return neg_n,pos_n,neut_n

#%%
texts = read_crawler_csv('')
#%%
ta = TweetsAnalysis()
preds = ta.predict(texts[2])
n,p,m = parse_preds(preds)
total = n+p+m
n/total
p/total
m/total




