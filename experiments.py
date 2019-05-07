# -*- coding:UTF-8 -*-

from pred import TweetsAnalysis
from numpy import genfromtxt
import pandas as pd

def read_crawler_csv(fname=''):
    return pd.read_csv(fname)

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
pf = read_crawler_csv('./results/tweeter_process.csv')
texts = pf.iloc[:,2]
#%%
ta = TweetsAnalysis()
preds = ta.predict(texts)
n,p,m = parse_preds(preds)
total = n+p+m
n/total,p/total,m/total
#%%
texts_11 = pf[pf.iloc[:,1]>'2018-11-01 00:00:00']
texts_11 = texts_11[texts_11.iloc[:,1]<'2018-11-31 24:00:00'].iloc[:,2]
preds = ta.predict(texts_11)
n,p,m = parse_preds(preds)
total = n+p+m
n/total,p/total,m/total





