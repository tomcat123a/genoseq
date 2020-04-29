# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 10:41:56 2020

@author: Administrator
"""

import os
import pandas as pd
os.chdir('C:/Users/Administrator/Desktop/QTLnet/result')

def cal_mean_sd(gene_idx,dnn,ety,cl,seed):
    filename= 'gene_{}_sed_{}_dnn_{}_ety{}_cl{}'.format(gene_idx,\
   seed,dnn,ety,cl)
    a=pd.read_csv(filename+'.csv')
    a['testcor']
        
os.listdir()
gene_idx=0
seed=4
dnn=10
ety=1
cl=1
