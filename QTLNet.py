# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 00:15:31 2020

@author: Administrator
"""


from ResNet1d import _resnet1d,BasicBlock,Bottleneck,Fc
from transformer import _transformer
from hyperopt import hp,tpe,Trials,fmin,STATUS_OK#pip install hyperopt --user
from torch.nn import Sequential ,MSELoss
import torch
from sklearn.model_selection import KFold
import numpy as np
class QTLNet():
    def __init__(loader=None):
        self.seed=seed
        self.maxepochs = maxepochs 
        self.earlystop= earlystop 
        self.verbose= verbose 
        self.loader=loader
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        
    def fit(self,train_x,train_y,val_x,val_y,bin_size,lr,batch_size,with_gap,earlystop,verbose):
        self.batch_size=batch_size
        optimizer=torch.optim.Adam(self.model.parameters(), lr=lr,amsgrad=True)
        loss_list=[]
        for i in range(self.maxepochs):
            optimizer.zero_grad()
            x,y=self.loader.get_batch()
            x=x.to(self.device)
            y=y.yo(self.device)
            
            ypred=self.model(x)
            
                    
            if ypred.dim()==2:
                ypred=ypred.squeeze(1)
            assert ypred.size()==y.size()
            loss =MSELoss(reduction='mean')(ypred,y)
            loss.backward()
            optimizer.step() 
            if earlystop==True:
                loss_list=loss_list.append( self.evaluate( val_x, val_y) )
            else:
                if train_x.size()[0]==batch_size:
                    loss_list=loss_list.append(loss.cpu().data.numpy())
                    
            if len(loss_list)>5 \
            and abs(loss_list[-2]/loss_list[-1]-1)<0.0001  :
                break
        if self.earlystop==True:
            return None,loss_list[-1]
        else:
            return loss_list[-1],self.evaluate( val_x, val_y)
    def step(data_x,data_y):
        pass
    def evaluate(self,data_x,data_y,batch_size,gene=None):
         ypred = self.model(data_x)
         if ypred.dim()==2:
             ypred=ypred.squeeze(1)
         assert ypred.size()==data_y.size()
         totalloss=0
         for i in range(np.ceil( int( data_x.size()[0]/batch_size))):
             loss = MSELoss(reduction='sum')(ypred,data_y).cpu().data.numpy()
             totalloss=totalloss+loss
         totalloss=totalloss/data_x.size()[0]
         return totalloss
         
         
    def cv(self,data_x,data_y):
        kf=KFold(n_splits=5, random_state=None, shuffle=False)
        error_list=[]
        for train_idx,test_idx in kf.split(data_x):
            self.init_model()
            _,valerror=self.fit( data_x[train_idx],data_y[train_idx] ,data_x[test_idx],data_y[test_idx])
            error_list.append(valerror) 
        return np.mean(error_list)        
    def init_model(self,param):
        conv=_resnet1d(block=BasicBlock if param['block_type']=='basic' else Bottleneck,\
    layers=[int(param['n_blocks_perlayer'])]*int(param['n_layers']),\
    channel_list=[param['first_channel']*2**i for i in range(param['n_layers'])],\
    block_stride=int(param['block_stride'])\
   ,conv_ker=int(param['cnn_kernel_size']),pool_ker=param['pool_ker'],\
   pool_stride=param['pool_stride'])
        out_features=int( param['first_channel']*2**(param['n_layers']-1) )
        dnn=Fc(inplanes=int( param['first_channel']*2**(param['n_layers']-1)),\
        hidden_list=[int( param['first_channel']*2**(param['n_layers']-1))]*int(param['dnn_depth']))
        self.model= Sequential(conv,dnn)
    def hyperselect(op='bayes'):
        
        if op=='bayes':
            space=[hp.quniform('cnn_kernel_size', 2, 20,1),#0\
         hp.quniform('pool_ker', 1, 10,1),#1\
         hp.quniform('block_stride', 1, 10,1),#2\
         hp.quniform('n_blocks_perlayer', 1, 3,1),#3\
        hp.quniform('n_layers',1, 4,1),#4\
        hp.choice('block', ['basic','bottleneck']),#5\
        hp.quniform('pool_stride',1, 4,1),#6\
        hp.quniform('cnn_stride',1,3,1),#7\
        hp.quniform('dnn_depth',1,2,1),#8\
    hp.quniform('with_gap',[True,False]),#9\
    hp.uniform('lr',0.0001,0.002),#10\
    hp.choice('first_channel',[4,8,16,32,64]),hp.quniform('bin_size',20,200,1)] 
             
            def obj(args):
                self.init_model(*args)
                self.fit(x=self.train_x,y=self.train_y,v_x=None,v_y=None,\
            bin_size=args[-1],lr=args[-3],with_gap=args[-4])
                return {'loss':loss, 'status': STATUS_OK, 'Trained_Model': 0}
                  
        tpe_best = fmin(fn=obj, space=space, 
                        algo=tpe_algo, trials=tpe_trials, 
                        max_evals=self.bayes_eval,rstate=np.random.RandomState(self.seed))
    def test():
        pass
    
class Loader():
    def __init__(self,batch_size,n,op):
        assert n>2*batch_size
        self.batch_size=batch_size
        self.n=n
        self.counter=0
        self.indices=torch.randperm(n ).split(batch_size)
    def get_item(self,i,datax,datay,gene=None):
        return datax[i],datay[i]
    def get_batch(self):
        if self.counter<self.n/self.batch_size:
            self.counter=self.counter+1
            
        else:
            self.counter=0
            self.indices=torch.randperm(n ).split(batch_size)
        return torch.cat([self.get_item(i) for i in self.indices[counter]],dim=0)
        
             
        
from hyperopt import hp,tpe,Trials,fmin,STATUS_OK

       
IUPAC_DICT = {
  'A' : torch.tensor([1, 0, 0, 0], dtype = torch.float32),
  'T' : torch.tensor([0, 1, 0, 0], dtype = torch.float32),
  'C' : torch.tensor([0, 0, 1, 0], dtype = torch.float32),
  'G' : torch.tensor([0, 0, 0, 1], dtype = torch.float32),
  '-' : torch.tensor([0, 0, 0, 0], dtype = torch.float32),
  '.' : torch.tensor([0, 0, 0, 0], dtype = torch.float32),
  'N' : torch.tensor([0.25, 0.25, 0.25, 0.25], dtype = torch.float32),
  'R' : torch.tensor([0.50, 0, 0, 0.50], dtype = torch.float32),
  'Y' : torch.tensor([0, 0.50, 0.50, 0], dtype = torch.float32),
  'S' : torch.tensor([0, 0, 0.50, 0.50], dtype = torch.float32),
  'W' : torch.tensor([0.50, 0.50, 0, 0], dtype = torch.float32),
  'K' : torch.tensor([0, 0.50, 0, 0.50], dtype = torch.float32),
  'M' : torch.tensor([0.50, 0, 0.50, 0], dtype = torch.float32),
  'B' : torch.tensor([0, 0.33, 0.33, 0.33], dtype = torch.float32),
  'D' : torch.tensor([0.33, 0.33, 0, 0.33], dtype = torch.float32),
  'H' : torch.tensor([0.33, 0.33, 0.33, 0], dtype = torch.float32),
  'V' : torch.tensor([0.33, 0, 0.33, 0.33], dtype = torch.float32)
}

 
def encodeSeqs(s, inputsize = 2000): 
    ''' Convert sequence to 0-1 encoding, forward and reverse complement sequence
    '''
     
    s=np.array(s[: inputsize ])
     
    codes = np.zeros((4, inputsize))
     
    for k, v in IUPAC_DICT.items(): 
        
        ix = np.where(s == k)[0]
        #ix = np.where(k == s)[0]
        if (len(ix) > 0): 
            codes[:, ix] = torch.unsqueeze(v, 1).expand(4, len(ix))
             
    return codes.astype("float32") 
      
    