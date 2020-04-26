# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 22:18:06 2020

@author: Administrator
"""

from patientseq import Patientwiseloader,Eval_single_gene
from QTLNet import QTLNet
from ResNet1d import CnnDnn
import torch

#Eval_single_gene(debug=1)
#Eval_single_gene(gene_idx=2,debug=1)
m=CnnDnn(dnn=[300,1])
x=torch.zeros(50,300,4,20)
m=CnnDnn(dnn=[300,1])
m1=CnnDnn(in_channel=4,out_channel=4,custom=True,dnn=[300,5,1])
print( m(x).size() )
print( m1(x).size())