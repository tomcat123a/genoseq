# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 22:18:06 2020

@author: Administrator
"""

from patientseq import Patientwiseloader,Eval_single_gene,fit,\
load_expression,split_train_test,encode_from_seq
from QTLNet import QTLNet,encodeSeqs
from ResNet1d import CnnDnn
import torch
import os
import numpy as np
'''
#os.chdir('C:/Users/Administrator/Desktop/QTLnet/')
Eval_single_gene(debug=1)
#Eval_single_gene(gene_idx=2,debug=1)
m=CnnDnn(dnn=[300,1])
x=torch.zeros(50,300,4,20)
m=CnnDnn(dnn=[300,1])
m1=CnnDnn(in_channel=4,out_channel=4,custom=True,dnn=[300,5,1])
print( m(x).size() )
print( m1(x).size())
'''
import pickle
ensemble_id='ENSG00000000457'
if os.name!='nt':
    pickle_file=open(\
 '/scratch/deepnet/dna/SNP2Expression/ciseQTLbin4Genes/{}.pkl'.format(ensemble_id),'rb')
else:
    pickle_file=open('data/ENSG00000000457.pkl','rb')
d=pickle.load(pickle_file)
pickle_file.close()
d=d.iloc[:,np.argsort(d.columns)]
#file=open('data/ciseQTL_chr21.pkl','rb')
#cis=pickle.load(file)
#file.close()
n_bins=d.shape[0]
m5=CnnDnn(in_channel=4,out_channel=4,custom=False,dnn=[n_bins,1])
m6=CnnDnn(in_channel=[4,4],out_channel=[4,4],cnn_ker=[3,3],\
          cnn_stride=[1,1],\
  pool_ker=[2,2],pool_stride=[2,2],custom=False,dnn=[n_bins,1])
 
tab=load_expression(tissue=2,\
   exp_folder='C:/Users/Administrator/Desktop/QTLnet/data/full_expr/'\
   if os.name=='nt' else '/scratch/deepnet/dna/full_expr/')
tab=tab.iloc[ :,np.argsort( tab.columns ) ]
d=d.iloc[ :,np.argsort( d.columns ) ]

train_x,train_y,test_x,test_y=split_train_test(d,tab.loc[ensemble_id])
train_x=encode_from_seq(train_x)
test_x=encode_from_seq(test_x)
train_y=torch.from_numpy( train_y.values.astype(np.float32) ) 
test_y=torch.from_numpy( test_y.values.astype(np.float32) ) 
fit(m5,train_x,train_y,maxepochs=100)
print('eval starts')
fit(m5,test_x,test_y,maxepochs=1)

 