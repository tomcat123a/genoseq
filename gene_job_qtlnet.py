# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 01:54:07 2020

@author: Administrator
"""
import os
os.chdir('C:/Users/Administrator/Desktop/QTLnet/jobfiles')
def genejob(jobname,txt):
    with open('{}.job'.format(jobname),'w') as file:
        L=['#BSUB -J {} \n'.format(jobname),\
        '#BSUB -o ../jobout/{}.out \n'.format(jobname),\
        '#BSUB -e ../jobout/{}.err \n'.format(jobname),\
        '#BSUB -n 1 \n',\
        '#BSUB -gpu "num=1" \n',\
        '#BSUB -q "normal" \n',\
        '#BSUB -W 36:00 \n',\
        '#BSUB -R "rusage[mem=16000]" \n',\
        'source /share/apps/ibm_wml_ce/1.6.1/anaconda3/etc/profile.d/conda.sh\n',\
        'conda activate wml_ce_env \n']
    
        for l in L:
            file.writelines(l)
        if isinstance(txt,list):
            for l in txt:
                file.writelines(l) 
        else:
            file.writelines(txt)
            
jobname_list=[]
for gene_idx in range(20):
    for seed in range(10):
        for dnn in [1,2,10]:
            
            filename='gene_{}_seed_{}_dnn_{}'.format(gene_idx,seed,dnn)
            jobname_list.append(filename+'.job')
            genejob(filename,'python /scratch/deepnet/qtlnet_xin_yl/pysrc/T.py --gene_idx {} --seed {} --dnn {}'.format(gene_idx,seed,dnn))
 
genejob('test','python /scratch/deepnet/qtlnet_xin_yl/pysrc/T.py --gene_idx {} --seed {} --dnn {} --epochs 2 '.format(0,2,1))                        

def subjob(jobname_list,subname):
    if not isinstance(jobname_list,list):
        jobname_list=[jobname_list]
    with open('sub{}.sh'.format(subname),'w') as file:
        for l in jobname_list:
            file.writelines('bsub < {} \n'.format(l))
            file.writelines('sleep 0.5s\n')  
            
subjob(jobname_list,'qtlnet')
