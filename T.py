# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 22:18:06 2020

@author: Administrator
"""

from patientseq import Patientwiseloader,Eval_single_gene,fit
from QTLNet import QTLNet,encodeSeqs
from ResNet1d import CnnDnn
import torch
import os
#os.chdir('C:/Users/Administrator/Desktop/QTLnet/')
Eval_single_gene(debug=1)
#Eval_single_gene(gene_idx=2,debug=1)
m=CnnDnn(dnn=[300,1])
x=torch.zeros(50,300,4,20)
m=CnnDnn(dnn=[300,1])
m1=CnnDnn(in_channel=4,out_channel=4,custom=True,dnn=[300,5,1])
print( m(x).size() )
print( m1(x).size())

import pickle
pickle_file=open('data/ENSG00000000457.pkl','rb')
d=pickle.load(pickle_file)
pickle_file.close()
d.iloc[1,0]
d.shape
file=open('data/ciseQTL_chr21.pkl','rb')
cis=pickle.load(file)
file.close()
m5=CnnDnn(in_channel=4,out_channel=4,custom=True,dnn=[d.shape[0],5,1])
train_x=torch.cat( [torch.stack(\
   [torch.from_numpy(encodeSeqs(list(d.iloc[row,i]))) \
  for row in range(d.shape[0])],dim=0).unsqueeze(0) \
   for i in range(60)],dim=0)
test_x=torch.cat( [torch.stack(\
   [torch.from_numpy(encodeSeqs(list(d.iloc[row,i]))) \
  for row in range(d.shape[0])],dim=0).unsqueeze(0) \
   for i in range(60,85)],dim=0)
fit(m5,train_x,train_y)

evaluate(m5,train_x,train_y)