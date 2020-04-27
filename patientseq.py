# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 20:55:45 2020

@author: Administrator
"""

from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import h5py
import torch  
import pyfasta
from torch.nn import Conv1d, ModuleList ,BatchNorm1d,Sequential,\
AdaptiveAvgPool1d,Linear,MSELoss,LSTM,GRU,MaxPool1d,AdaptiveMaxPool1d,AvgPool1d
from scipy.stats import pearsonr as cor
import time

def load_df(file_name):
    pickle_file=open(file_name,'rb')
    d=pickle.load(pickle_file)
    pickle_file.close()
    return d

def split_train_test(d,train_percent=0.8):
    n=d.shape[1]
    perm_idx=np.random.permutation(n)
    train_idx=perm_idx[:int(n*train_percent)]
    test_idx=perm_idx[int(n*train_percent):]
    return d.iloc[:,train_idx],d.iloc[:,test_idx]
def fit(model,data_x,data_y,lr=0.001,maxepochs=100,\
   device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),debug=1):
    optimizer=torch.optim.Adam(model.parameters(), lr=lr,amsgrad=True)
    loss_list=[]
    cor_list=[]
    model=model.train().to(device)
    for i in range( maxepochs):
        optimizer.zero_grad()
        x,y=data_x.to(device),data_y.to(device)
        ypred=model(x)
        if ypred.dim()==2:
            ypred=ypred.squeeze(1)
        assert ypred.size()==y.size()
        loss =MSELoss(reduction='mean')(ypred,y)
        loss.backward()
        optimizer.step() 
        loss_list=loss_list.append(loss.cpu().data.numpy())
        cor_list=cor_list.append(cor(y.cpu().data.numpy(),ypred.cpu().data.numpy())[0]) 
        if debug==1 and maxepochs>1:
            print(loss_list[-1],cor_list[-1])
        scipy.stats.spearmanr()
        if len(loss_list)>5 \
        and abs(loss_list[-2]/loss_list[-1]-1)<0.0001  :
            break
    print(loss_list[-1],cor_list[-1])

def cv(data_x,data_y):
    kf=KFold(n_splits=5, random_state=None, shuffle=False)
    rerro_list=[]
    for train_idx,test_idx in kf.split(data_x):
        self.init_model()
        _,valerror=self.fit( data_x[train_idx],data_y[train_idx] ,data_x[test_idx],data_y[test_idx])
        error_list.append(valerror) 
    return np.mean(error_list)            
    
def Eval_single_gene(folder=\
   '/scratch/deepnet/dna/SNP2Expression/ciseQTLbin4Genes/',\
    gene_idx=0,debug=1):
    l_gene=os.listdir(folder)
    d=load_df(folder+l_gene[gene_idx])
    if debug:
        print(d.shape)
    test_error=0
    train_d,test_d=split_train_test(d)
    if debug:
        print(train_d.shape)
        print(test_d.shape)
        print(test_d)
        print(len(test_d.iloc[0,0]))
    return test_error
    

class Patientwiseloader():
    """return both the training and test dataset."""

    def __init__(self,  tissue ,chrom ,\
                 seq_folder='/home/yxz346/dna20191127/data/DeepSeQ3/',\
                 exp_folder='/scratch/deepnet/dna/full_expr/'\
,start=1,end=3/7  ):
        """
        Args:
            tissue:values:0,1,2,3, ('Breast_Mammary_Tissue','Muscle_Skeletal','Ovary','Whole_Blood')
            exp_type:values: 0,1,2, 
            np (float) : percentage of number of patients selected.
            chrom in [6,8]
            len_per (float) :percentage of dna sequences selected
            n_train_gene_rate(float): ratio of the genes that are in the training data set, the 
            #rest are in the test data set.
            #seq folder '/home/yilun/dna/seqdata/chr8'
            #read seq data a = pd.read_csv('patient_id.txt',sep='\t') a.iloc[i,1] the first column at row i,
            #a.iloc[i,2] the second column at row i,a.iloc[:,0] is the gene-name list.
            #exp_folder '/home/yilun/dna/exp_nor_rbe/','NORM_RM.txt',
            #'/home/yilun/dna/exp_unnor_rbe/' ,'UNNORM_RM.txt', '/home/yilun/dna/exp_raw/','RAW.txt'
            #read expdata b = pd.read_csv('name.txt',sep='\t') b.iloc[i,0] is the expression value for
            first patient at gene i,
             b.index is the gene-name list,which is the same as a.iloc[:,0].b.columns is the list
             patient_id. The patients in this list will be fetched for the corresponding dna sequence in
             seq folder /home/yilun/dna/seqdata/SAMPLE2GENOTYPE.DICT
             
            #traindataset=1output training dataset
            #=2,output val dataset
            #=3,output test dataset
             Note:
                 1.all pd.read_csv must have sep='\t'
                 
            rev_train: 0 plain 1 augmented 2 channel concatenation 3 shape concatenation
        """
        #first columne of dicttable, is column names of exp_table
        #second columne of dicttable, is h5.keys()
        #os.listdir() files for seq
         
        
             
         
        self.start=start
        self.end=end
         
         
            
        self.TISSUE=['Breast_Mammary_Tissue','Muscle_Skeletal','Ovary','Whole_Blood',\
                'Adipose_Subcutaneous','Brain_Hippocampus','Esophagus_Mucosa','Prostate','Adipose_Visceral_Omentum','Brain_Hypothalamus'\
                ,'Skin_Not_Sun_Exposed_Suprapubic','Esophagus_Muscularis','Adrenal_Gland','Brain_Nucleus_accumbens_basal_ganglia','Fallopian_Tube',\
                'Skin_Sun_Exposed_Lower_leg','Artery_Aorta','Brain_Putamen_basal_ganglia','Heart_Atrial_Appendage','Small_Intestine_Terminal_Ileum',\
                'Artery_Coronary','Brain_Spinal_cord_cervical_c-1','Heart_Left_Ventricle','Spleen','Artery_Tibial','Brain_Substantia_nigra',\
                'Kidney_Cortex','Stomach','Bladder','Liver','Testis',\
                'Brain_Amygdala','Cells_EBV-transformed_lymphocytes','Lung','Thyroid','Brain_Anterior_cingulate_cortex_BA24','Cells_Transformed_fibroblasts',\
                'Minor_Salivary_Gland','Uterus','Brain_Caudate_basal_ganglia','Cervix_Ectocervix','Muscle_Skeletal','Vagina',\
                'Brain_Cerebellar_Hemisphere','Cervix_Endocervix','Nerve_Tibial','Brain_Cerebellum','Colon_Sigmoid','Brain_Cortex','Pancreas',\
                'Colon_Transverse','Brain_Frontal_Cortex_BA9','Esophagus_Gastroesophageal_Junction','Pituitary']
        
        
        
        self.tissue_num=tissue 
        self.tissue=self.TISSUE[tissue]
        print(self.tissue+' selected')
        exp_folder_dir_list=[]
        for j in chrom:
            #EXP_FOLDER= exp_folder+self.TISSUE[tissue]+'/chr{}_RAW.csv'.format(j) in avg
            EXP_FOLDER= exp_folder+'chr{}/exp_nor_rbe/'.format(j)+self.TISSUE[tissue]+'/NORM_RM.txt' #in avg_0
            exp_folder_dir_list.append(EXP_FOLDER)
        if os.name!='nt':#combine chrs of expression,
            exp_table=pd.concat([ pd.read_csv(exp_folder_dir_list[j],index_col=0,sep='\t',engine='python') for j in range(len(chrom))])
        else:
            exp_table=pd.read_csv('C:/tmp/NORM_RM.txt',index_col=0,sep='\t') 
        #print(exp_table )
        
        #self.tissue_patlist=pd.read_csv('/scratch/deepnet/dna/allreplace_xin/ovary_patients_idx.csv').iloc[:,0]
        self.tissue_patlist=get_patidx_new2(tis=self.tissue,folder='/scratch/deepnet/dna/patvcf/')
        self.pat_num=exp_table.shape[1]
        self.gene_snp_cached_df={} 
        #nonsig_eqtldf=pd.read_csv('nonsigsnp_ovary.csv' )
        #nonsig_eqtldf=pd.read_csv('/scratch/deepnet/dna/VCF/eQTL/nonsigsnp_ovary.txt', header = None)
        eqtldf=pd.read_csv('/scratch/deepnet/dna/VCF/eQTL/Ovary_Analysis.v6p.signif_snpgene_pairs.txt', \
                           sep = '\t')
        #eqtldf=pd.read_csv('/scratch/deepnet/dna/VCF/eQTL/Ovary_Analysis.v6p.all_snpgene_pairs.txt', header = None)
        self.sig_gene_id=[x.split('.')[0] for x in eqtldf['gene_id']]
         
        #filter out low-expressed genes
        #exp_table=exp_table[exp_table>exp_table.quantile(0.2)].dropna()
        
        self.geneinfo=pd.read_csv('/scratch/deepnet/dna/allreplace_xin/gene_chr_tss_stra.csv').set_index('geneid')
        self.gene_ensemble_id=np.intersect1d( list( exp_table.index ),list( self.geneinfo.index)  )
        self.gene_num=len(self.gene_ensemble_id)
        #geneinfo=pd.read_csv('gene_chr_tss_stra.csv',index_col=0)
        #print('num_genes :{}'.format(exp_table.shape[0]))
        #print('num_pat :{}'.format(exp_table.shape[1]))
        self.exp_shape=[exp_table.shape[0],exp_table.shape[1]]
         
         
        hg19_fasta = "/scratch/deepnet/dna/VCF/hg19.fa"
        #hg19_fasta='C:/Users/Administrator/Desktop/seqex/pysrc/hg19/hg19.fa'
        self.Genome = pyfasta.Fasta(hg19_fasta)
        
        #self.dfy =  torch.from_numpy( exp_table.values.reshape(-1,order='F').astype(np.float32) ).type(torch.float32)
         
        
        #this is important ,only the second segment is the patient name
        exp_table.columns=[x.split('-')[1] for x in exp_table.columns]
        self.exp_table_save=exp_table
        
        self.dfy = torch.from_numpy( exp_table.values.astype(np.float32) ).type(torch.float32)  
        
        self.gene_keys= list(exp_table.index) 
         
        
        #print('choosen={},unchoosen{}'.format(choosen,unchoosen))
           
        uplength=70000
        self.left=int(uplength*start)
        self.right=int(uplength*end)
        print('start site :{},end site :{},len :{}'.format(uplength-int(uplength*self.start),uplength+int(uplength*self.end),\
              1+int(uplength*start)+int(uplength*end) ))
        self.dna_len=1+int(uplength*start)+int(uplength*end)
        idx = torch.tensor( [i for i in range(int(self.dna_len) , -1, -1)] )
        self.idx = idx
        #geneinfo_pickle_file= open('C:/Users/Administrator/Desktop/seqex/pysrc/config/gene4inf.pkl' ,'rb') 
        #geneinfo_pickle_file= open('/scratch/deepnet/dna/allreplace_xin/gene4inf.pkl' ,'rb') 
        #self.geneinfo=pickle.load(geneinfo_pickle_file)
        #geneinfo_pickle_file.close()
        
        self.chr_cached={}
    def __len__(self):
        return len(self.dfy)
         
    def generate_split_idx(self,train_percent,val_percent):
        n=self.exp_table_save.shape[0]
        np.random.seed(self.seed)
        resample=np.random.permutation(n)
        train_n=int(n*train_percent)
        val_n=int( n*val_percent )
        di={}
        di['train']=resample[:train_n]
        di['val']=resample[train_n:train_n+val_n]
        di['test']=resample[train_n+val_n:]
        return di
        
    def save_txt(self,ensemble_idx_list,savedir):
        for gene_ensemble_id in ensemble_idx_list:
            chr_num = self.geneinfo.loc[gene_ensemble_id]['chrnum']
            if not chr_num in self.chr_cached.keys():
                #os.chdir('C:/Users/Administrator/Desktop/seqex/pysrc/config')
                pickle_file= open('/scratch/deepnet/dna/allreplace_xin/vcf_chr{}.pkl'.format(chr_num),'rb') 
                #pickle_file= open('vcf_chr{}.pkl'.format(chr_num),'rb') 
                chr_eqtl_pat=pickle.load(pickle_file)
                #each value is 0,1,2
                #change column name to patient name
                chr_eqtl_pat.columns=[x.split('-')[1] for x in chr_eqtl_pat.columns]
                self.chr_cached[str(chr_num)]=chr_eqtl_pat
            else:
                chr_eqtl_pat=self.chr_cached[str(chr_num)]
            pickle_file.close()
            #take patient belonging to this tissue    pat
            #chr_eqtl_pat=chr_eqtl_pat[exp_table.columns]
            chr_eqtl_pat=chr_eqtl_pat[self.exp_table_save.columns]
            if not gene_ensemble_id in self.gene_snp_cached_df.keys():
                #gene2snp_pickle_file= open('gene2snp.pkl' ,'rb') 
                gene2snp_pickle_file= open('/scratch/deepnet/dna/allreplace_xin/gene2snp.pkl' ,'rb') 
                gene2snp=pickle.load(gene2snp_pickle_file)
                snps_in_gene=gene2snp[gene_ensemble_id]
                #a list of snps names by  '21_45743594_C_T_b37' chr_loc_ref_alt_b37
                #pure it to remove duplicate first 2 items
                #pickle_mf=open('C:/Users/Administrator/Desktop/seqex/pysrc/config/UniqueSNPfilterByMAF_max.pkl','rb') 
                pickle_mf=open('/scratch/deepnet/dna/allreplace_xin/UniqueSNPfilterByMAF_max.pkl','rb') 
                
                maf=pickle.load(pickle_mf)
                pickle_mf.close()
                chr_maf=maf['chr{}'.format(chr_num)]
                snps_in_gene=[ chr_maf[x.split('_')[0]+'_'+x.split('_')[1]] for x in snps_in_gene ]
                snps_in_gene=list(set(snps_in_gene))
                self.gene_snp_cached_df[gene_ensemble_id]=snps_in_gene
                gene2snp_pickle_file.close() 
            else:
                snps_in_gene=self.gene_snp_cached_df[gene_ensemble_id] 
            chr_eqtl_pat.index=chr_eqtl_pat.index.astype(str)
             
            chr_eqtl_pat.loc[snps_in_gene].to_csv('{}/{}_snp_pat.csv'.format(savedir,gene_ensemble_id))
            self.exp_table_save.loc[gene_ensemble_id].to_csv('{}/{}_expr.csv'.format(savedir,gene_ensemble_id))
             
    def __getitem__(self, pat,gene_ensemble_id,binwise=True,one_side_flank=6,op='onlyone' ):
        assert pat in range(self.pat_num)
        #ENSG00000141956 strand -1
        #ENSG00000141959 strand 1
        #ENSG00000004139 strand 1 chr17
        #ENSG00000004139 strand -1 chr17
        #gene_ensemble_id='ENSG00000141956'
        #gene_ensemble_id='ENSG00000141959'
        #gene_ensemble_id='ENSG00000004139'
        #gene_ensemble_id='ENSG00000004142'
         
        assert gene_ensemble_id in self.geneinfo.index
         
        chr_num = self.geneinfo.loc[gene_ensemble_id]['chrnum']
        if not chr_num in self.chr_cached.keys():
            #os.chdir('C:/Users/Administrator/Desktop/seqex/pysrc/config')
            pickle_file= open('/scratch/deepnet/dna/allreplace_xin/vcf_chr{}.pkl'.format(chr_num),'rb') 
            #pickle_file= open('vcf_chr{}.pkl'.format(chr_num),'rb') 
            chr_eqtl_pat=pickle.load(pickle_file)
            #each value is 0,1,2
            #change column name to patient name
            chr_eqtl_pat.columns=[x.split('-')[1] for x in chr_eqtl_pat.columns]
            self.chr_cached[str(chr_num)]=chr_eqtl_pat
        else:
            chr_eqtl_pat=self.chr_cached[str(chr_num)]
        pickle_file.close()
        #take patient belonging to this tissue    pat
        #chr_eqtl_pat[exp_table.columns]
        chr_eqtl_pat=chr_eqtl_pat[self.exp_table_save.columns]
        if not gene_ensemble_id in self.gene_snp_cached_df.keys():
            #gene2snp_pickle_file= open('gene2snp.pkl' ,'rb') 
            gene2snp_pickle_file= open('/scratch/deepnet/dna/allreplace_xin/gene2snp.pkl' ,'rb') 
            gene2snp=pickle.load(gene2snp_pickle_file)
            snps_in_gene=gene2snp[gene_ensemble_id]
            #a list of snps names by  '21_45743594_C_T_b37' chr_loc_ref_alt_b37
            #pure it to remove duplicate first 2 items
            #pickle_mf=open('C:/Users/Administrator/Desktop/seqex/pysrc/config/UniqueSNPfilterByMAF_max.pkl','rb') 
            pickle_mf=open('/scratch/deepnet/dna/allreplace_xin/UniqueSNPfilterByMAF_max.pkl','rb') 
            
            maf=pickle.load(pickle_mf)
            pickle_mf.close()
            chr_maf=maf['chr{}'.format(chr_num)]
            snps_in_gene=[ chr_maf[x.split('_')[0]+'_'+x.split('_')[1]] for x in snps_in_gene ]
            snps_in_gene=list(set(snps_in_gene))
            self.gene_snp_cached_df[gene_ensemble_id]=snps_in_gene
            gene2snp_pickle_file.close() 
        else:
            snps_in_gene=self.gene_snp_cached_df[gene_ensemble_id] 
        pat_chr_snp_replace=chr_eqtl_pat.iloc[:,pat]
        #pat_chr_snp_replace a single column of 0,1,2,for this patient,pat_gene_snp_replace.index is its name
        pat_gene_snp_replace=pat_chr_snp_replace[snps_in_gene]
        gene_tss=self.geneinfo.loc[gene_ensemble_id]['tss']
        #gene_tss=geneinfo.loc[gene_ensemble_id]['tss']
        gene_strand=self.geneinfo.loc[gene_ensemble_id]['strand']
        #gene_strand=geneinfo.loc[gene_ensemble_id]['strand']
        if binwise==False:
            ref_seq_inlist,tss_loc_on_refseq=self.get_ref_seq_list(chr_num,gene_tss,self.left ,\
                self.right ,gene_strand)
            replace_value_list=[2]
            alt_seq_inlist_11=self.post_replace_seq_list(chr_num,ref_seq_inlist,tss_loc_on_refseq,\
            pat_gene_snp_replace,replace_value_list,gene_strand)
            replace_value_list=[1,2]
            alt_seq_inlist_10=self.post_replace_seq_list(chr_num,ref_seq_inlist,tss_loc_on_refseq,\
            pat_gene_snp_replace,replace_value_list,gene_strand)
            
            pickle_file.close()
            pat_seq_11=self.encodeSeqs(alt_seq_inlist_11,self.left+self.right+1)
            pat_seq_10=self.encodeSeqs(alt_seq_inlist_10,self.left+self.right+1)
            
            return torch.from_numpy(pat_seq_11) ,torch.from_numpy(pat_seq_10),\
       torch.from_numpy(np.array(self.exp_table_save.loc[gene_ensemble_id][pat]))
        else:
            bins=self.get_bins_replaced(chr_num,gene_strand,\
        pat_gene_snp_replace,one_side_flank=one_side_flank,op=op)
            return torch.stack([torch.from_numpy( \
   self.encodeSeqs(x,2*one_side_flank+1)) for x in bins],dim=0),\
            torch.from_numpy(np.array(self.exp_table_save.loc[gene_ensemble_id][pat]))
   
    def available_ensemble_id(self,idx=None):
        if idx is None:
            print( list(self.geneinfo.index ) )
        else:
            print( np.array(self.geneinfo.index )[idx] )
            return np.array(self.geneinfo.index )[idx] 
    def reverse(self,x):
         
        return x.index_select(1,self.idx).index_select(0,torch.tensor([1,0,3,2]))
    def flip(self,x):
        return x.index_select(0,torch.tensor([1,0,3,2]))
    def reverse_0(self,x,device ):
        return x.index_select(2,torch.tensor(list(range(x.size()[-1]-1,-1,-1))).to(device)).index_select(1,torch.tensor([1,0,3,2]).to(device))
    def reverse_compli(self,string_of_ATCG):
        def rever_list(character):
            if character=='A':
                return 'T'
            if character=='T':
                return 'A'
            if character=='C':
                return 'G'
            if character=='G':
                return 'C'
            else:
                return character
        ret=''
        out=ret.join( [rever_list(x) for x in string_of_ATCG] )
        
        return out[::-1]
            
    def take_patients(self,file):
        pass
    
    def get_bins_replaced(self,chr_num,strand,pat_gene_snp_replace,\
        one_side_flank=6,op='onlyone'):
        snp_bins=[]
        locs=[int(x.split('_')[1]) for x in pat_gene_snp_replace.index]
        
        true_flank=one_side_flank
        one_side_flank=one_side_flank+10
        for i in range(len(pat_gene_snp_replace.index)):
            if op=='all':
                i_list=[i]
                for forward in range(one_side_flank):
                    if i-forward<0:
                        break
                    if abs(locs[i-forward]-locs[i])<=one_side_flank and \
                    pat_gene_snp_replace[i-forward]!=0:
                        i_list.append(i-forward)
                    if abs(locs[i-forward]-locs[i])>one_side_flank:
                        break
                for backward in range(one_side_flank):
                    if i+backward>=len(pat_gene_snp_replace.index)-1 :
                        break
                    if abs(locs[i+backward]-locs[i])<=one_side_flank and \
                     pat_gene_snp_replace[i+backward]!=0:
                        i_list.append(i+backward)
                    if abs(locs[i-backward]-locs[i])>one_side_flank:
                        break
                snp_loc=locs[i]
                start=snp_loc-one_side_flank
                end=snp_loc+one_side_flank
                ref_list=list( self.Genome.sequence({'chr': 'chr{}'.format(chr_num), \
        'start':start, 'stop': end, 'strand': strand}).upper())
                replace_value_list=[2]
                 
                snp_bins.append(self.post_replace_seq_list(chr_num,ref_list,snp_loc,np.take(\
            pat_gene_snp_replace,i_list),replace_value_list,strand,True,true_flank) )
                replace_value_list=[1,2]
                snp_bins.append(self.post_replace_seq_list(chr_num,ref_list,snp_loc,np.take(\
            pat_gene_snp_replace,i_list),replace_value_list,strand,True,true_flank))
            else:
                snp_loc=locs[i]
                start=snp_loc-one_side_flank
                end=snp_loc+one_side_flank
                ref_list=list( self.Genome.sequence({'chr': 'chr{}'.format(chr_num), \
            'start':start, 'stop': end, 'strand': strand}).upper())
                name=pat_gene_snp_replace.index[i]
                 
                if pat_gene_snp_replace[i]==0:
                    snp_bins.append( ref_list[one_side_flank-true_flank:one_side_flank+true_flank] )
                    snp_bins.append( ref_list[one_side_flank-true_flank:one_side_flank+true_flank] )
                else:
                    old_qtlseq_len=len( name.split('_')[2] )
                    new_qtlseq=name.split('_')[3]
                    a=ref_list.copy()
                    a[one_side_flank:one_side_flank+old_qtlseq_len]=new_qtlseq
                    assert old_qtlseq_len - len(new_qtlseq) <=one_side_flank-true_flank
                    if pat_gene_snp_replace[i]==1:
                        snp_bins.append( a[one_side_flank-true_flank:one_side_flank+true_flank] )
                        snp_bins.append( ref_list[one_side_flank-true_flank:one_side_flank+true_flank] )
                    else:
                        snp_bins.append( a[one_side_flank-true_flank:one_side_flank+true_flank] )
                        snp_bins.append( a[one_side_flank-true_flank:one_side_flank+true_flank] )
        return snp_bins
    
    def get_ref_seq_list(self,chr_num,TSS,left,right,strand,for_bin=False):
        if for_bin==False:
            max_len=max(left,right)+5000
            return list( self.Genome.sequence({'chr': 'chr{}'.format(chr_num), \
    'start': TSS - max_len, 'stop': TSS + max_len, 'strand': strand}).upper()) ,max_len
        else:
            max_len=max(left,right)+50
            return list( self.Genome.sequence({'chr': 'chr{}'.format(chr_num), \
    'start': TSS - max_len, 'stop': TSS + max_len, 'strand': strand}).upper()) 
        #return list(self.Genome.sequence({'chr': 'chr{}'.format(chr_num), 'start': TSS - left, 'stop': TSS + right, 'strand': strand}).upper())
    
    def post_replace_seq_list(self,chr_num,ref_seq_list,ref_seq_tss_org_loc,pat_gene_snp_replace,\
  replace_value_list,strand,for_bin=True,true_flank=10):
        new_seq_list=ref_seq_list.copy()
        #replace_value_list is [1] or [1,2]
        bias=0
        sum_of_bias=0
        current_tss=ref_seq_tss_org_loc
         
        for qtl in pat_gene_snp_replace.index:
            qtl_loc_org=int(qtl.split('_')[1])
            qtl_occurences=pat_gene_snp_replace.loc[qtl]
            if type(qtl_occurences)!=np.int64:
                qtl_occurences=qtl_occurences.iloc[0]
            if not ( qtl_occurences in replace_value_list ):
                continue
             
            qtl_current_loc=ref_seq_tss_org_loc+strand*(qtl_loc_org-ref_seq_tss_org_loc)+sum_of_bias
            if strand==1:
                qtl_new_value=list( qtl.split('_')[3] if strand==1 else self.reverse_compli(qtl.split('_')[3]) )
            else:
                rev_compli=self.reverse_compli(\
    list( qtl.split('_')[3] if strand==1 else self.reverse_compli(qtl.split('_')[3]) ) )
                 
                qtl_new_value=rev_compli
            
            qtl_old_len=len(qtl.split('_')[2])
            bias=len(qtl_new_value)-qtl_old_len
            sum_of_bias=sum_of_bias+bias
            if qtl_current_loc<current_tss:
                current_tss=current_tss+bias
            new_seq_list[qtl_current_loc:qtl_current_loc+qtl_old_len]=qtl_new_value
        if for_bin==False:
            assert  current_tss>=self.left and len(new_seq_list)>=self.right+ current_tss
            return new_seq_list[ current_tss-self.left: current_tss+self.right]
        else:
            return new_seq_list[ current_tss-true_flank: current_tss+true_flank]
    
    def replace(self,x,genekey,pat):
        pass
    
    def encodeSeqs(self,s, inputsize = 2000): 
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

        s=np.array(s[: inputsize ])
         
        codes = np.zeros((4, inputsize))
         
        for k, v in IUPAC_DICT.items(): 
            
                
            ix = np.where(s == k)[0]
            #ix = np.where(k == s)[0]
            if (len(ix) > 0): 
                codes[:, ix] = torch.unsqueeze(v, 1).expand(4, len(ix))
                 
        return codes.astype("float32")
    
    def __get_single_snprepla_item__(self, pat,gene_ensemble_id):
        assert pat in range(self.pat_num)
        #ENSG00000141956 strand -1
        #ENSG00000141959 strand 1
         
        #gene_ensemble_id='ENSG00000141956'
        #gene_ensemble_id='ENSG00000141959'
        chr_num = self.geneinfo.loc[gene_ensemble_id]['chrnum']
        pickle_file= open('/scratch/deepnet/dna/allreplace_xin/vcf_chr{}.pkl'.format(chr_num),'rb') 
        #pickle_file= open('vcf_chr{}.pkl'.format(chr_num),'rb') 
        chr_eqtl_pat=pickle.load(pickle_file)
        #take patient belonging to this tissue
        chr_eqtl_pat.columns=[x.split('-')[1] for x in chr_eqtl_pat.columns]
        
        chr_eqtl_pat=chr_eqtl_pat[self.exp_table_save.columns]
        if not gene_ensemble_id in self.gene_snp_cached_df.keys():
            gene2snp_pickle_file= open('/scratch/deepnet/dna/allreplace_xin/gene2snp.pkl' ,'rb') 
            gene2snp=pickle.load(gene2snp_pickle_file)
            snps_in_gene=gene2snp[gene_ensemble_id]
            self.gene_snp_cached_dict[gene_ensemble_id]=snps_in_gene
            #pure it to remove duplicate first 2 items
            #pickle_mf=open('C:/Users/Administrator/Desktop/seqex/pysrc/config/UniqueSNPfilterByMAF_max.pkl','rb') 
            pickle_mf=open('/scratch/deepnet/dna/allreplace_xin/UniqueSNPfilterByMAF_max.pkl','rb') 
            
            maf=pickle.load(pickle_mf)
            pickle_mf.close()
            chr_maf=maf['chr{}'.format(chr_num)]
            snps_in_gene=[ chr_maf[x.split('_')[0]+'_'+x.split('_')[1]] for x in snps_in_gene ]
            gene2snp_pickle_file.close() 
        else:
            snps_in_gene=self.gene_snp_cached_dict[gene_ensemble_id] 
        pat_chr_snp_replace=chr_eqtl_pat.iloc[:,pat]
        pat_gene_snp_replace=pat_chr_snp_replace[snps_in_gene]
        gene_tss=self.geneinfo.loc[gene_ensemble_id]['tss']
        #gene_tss=geneinfo.loc[gene_ensemble_id]['tss']
        gene_strand=self.geneinfo.loc[gene_ensemble_id]['strand']
        #gene_strand=geneinfo.loc[gene_ensemble_id]['strand']
        ref_seq_inlist,tss_loc_on_refseq=self.get_ref_seq_list(chr_num,gene_tss,self.left ,self.right ,gene_strand)
        replace_value_list=[2]
        alt_seq_inlist_11=self.post_replace_seq_list(chr_num,ref_seq_inlist,tss_loc_on_refseq,\
        snps_in_gene,pat_gene_snp_replace,replace_value_list,gene_strand)
        replace_value_list=[1,2]
        alt_seq_inlist_10=self.post_replace_seq_list(chr_num,ref_seq_inlist,tss_loc_on_refseq,\
        snps_in_gene,pat_gene_snp_replace,replace_value_list,gene_strand)
        
        pickle_file.close()
        pat_seq_11=self.encodeSeqs(alt_seq_inlist_11,self.left+self.right+1)
        pat_seq_10=self.encodeSeqs(alt_seq_inlist_10,self.left+self.right+1)
        
        return torch.from_numpy(pat_seq_11) ,torch.from_numpy(pat_seq_10),\
   torch.from_numpy(np.array(self.exp_table_save.loc[gene_ensemble_id][pat]))    
    
def tissuevcf(tis='Ovary',folder='/scratch/deepnet/dna/patvcf/',mode=1):
    for Chr in list(range(1,23)):
        
        exp_folder='/scratch/deepnet/yxz346/tmp/data/data/RawExpression/'
        expressionfile=exp_folder+'chr{}/exp_raw/'.format(Chr)  +'{}/'.format(tis)+'RAW.txt'  
        exp_table=pd.read_csv(expressionfile,index_col=0,sep='\t' )
        patientnames=exp_table.columns
        sample = pd.read_csv("/scratch/deepnet/dna/VCF/GTeXix.csv", header = None)
        if mode==1:
            #eqtldf=pd.read_csv('Ovary_Analysis.v6p.signif_snpgene_pairs.txt', sep = '\t')
        
            eqtldf=pd.read_csv('/scratch/deepnet/dna/VCF/eQTL/Ovary_Analysis.v6p.signif_snpgene_pairs.txt', sep = '\t', header = None)
        else:
            #obtained by 
            #p1=pd.read_csv('Ovary_Analysis.v6p.all_snpgene_pairs.txt',sep='\t')
            #p1.loc[p1['pval_nominal']>0.998].to_csv('nonsigsnp_ovary.txt',index=False)
            eqtldf=pd.read_csv('/scratch/deepnet/dna/VCF/eQTL/nonsigsnp_ovary.txt', header = None)
        
        if mode==1:
            eqtlnames=eqtldf.iloc[1:,0] #all ovary qtls
        else:
            eqtlnames=eqtldf.iloc[1:,1]
        eqtl_slope=eqtldf.iloc[1:,4]
        print(eqtlnames)
        chrnames= [ x  for x in eqtlnames if x.split('_')[0]==str(Chr)]
        slope= eqtl_slope[[x.split('_')[0]==str(Chr) for x in eqtlnames] ]
         
        print('chr{} starts'.format(Chr))
        vcf = pd.read_csv('/scratch/deepnet/dna/VCF/VCF/chr{}.vcf'.format(Chr), sep = '\t', header = None)
        newvcf=vcf.set_index(0)
        newvcf_ovary_pat=newvcf.take([0,1,2,3]+[i+4 for i in range(450) if sample.iloc[i,0].split('-')[1] in [y.split('-')[1] for y in patientnames] ],axis=1)
        vcf_inqtl=pd.DataFrame([ newvcf_ovary_pat.loc[x] for x in chrnames ])
        vcf_inqtl_re_id=vcf_inqtl.reset_index()
        vcf_inqtl_re_id.iloc[:,1]=slope.values 
        vcf_inqtl_re_id.to_csv(folder+'patvcf_{}.vcf'.format(Chr),index=False) 
        #to read,use the code: pd.read_csv('patvcf_21,vcf')
        #vcf_inqtl 0th col eqtl names, 1st col col, slope
        print('chr{} finished'.format(Chr))
        
def get_patidx(tis='Ovary',folder='/scratch/deepnet/dna/patvcf/',mode=1):
     
        
    expressionfile='/scratch/deepnet/dna/full_expr/chr1/exp_nor_rbe/Ovary/NORM_RM.txt'
     
    exp_table=pd.read_csv(expressionfile,index_col=0,sep='\t' )
    patientnames=exp_table.columns
    sample = pd.read_csv("/scratch/deepnet/dna/VCF/GTeXix.csv", header = None)
    l1=[i for i in range(450) if sample.iloc[i,0].split('-')[1] in [y.split('-')[1] for y in patientnames] ] 
    p=pd.DataFrame()   
    p['ovary']=l1
    print(p)
    p.to_csv('/scratch/deepnet/dna/allreplace_xin/ovary_patients_idx.csv',index=False)
    
def get_patidx_new2(tis='Ovary',folder='/scratch/deepnet/dna/patvcf/'):
     
        
    expressionfile='/scratch/deepnet/dna/full_expr/chr1/exp_nor_rbe/{}/NORM_RM.txt'.format(tis)
     
    exp_table=pd.read_csv(expressionfile,index_col=0,sep='\t' )
    patientnames=exp_table.columns
    sample = pd.read_csv("/scratch/deepnet/dna/VCF/GTeXix.csv", header = None)
    l1=[i for i in range(450) if sample.iloc[i,0].split('-')[1] in [y.split('-')[1] for y in patientnames] ] 
    return l1
        
def fwd(ref_x,alt_x,mod_file='C:/Users/Administrator/Desktop/seqex/pysrc/config/70019.pt'):
    from seq import testmlp
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    totallen=70000
    ARG_LEFT=1
    ARG_RIGHT=3/7
    params={'n_features':1+int(totallen*ARG_LEFT+totallen*ARG_RIGHT),'ker_size':[5]+[3]*6,'stride':[1]*5,\
                  'channels_list':[4]+[160,320,480,640,960],'dnn_list':[2,1],'di':[1,2,2,4,2] ,'res_on':1,\
                  'poolsize':[800,200,100,50,30],'ppoolstride':[200,8,2,2,2],'pooltype':3 }
    
    model=testmlp(**params)
    model.load_state_dict(torch.load(mod_file,map_location=device))
    model=model.to(device)
    tensor_ref=torch.from_numpy(ref_x).to(device)
    tensor_alt=torch.from_numpy(alt_x).to(device)
    alt_v=model(tensor_alt).data.cpu().numpy()
    ref_v=model(tensor_ref).data.cpu().numpy()
    return ref_v,alt_v

     
    
def train_model(model,loader,pat_idx_list,gene_idx,batch_size,epochs,verbose,lr,w2_lambda,device,combine_mode=2,seed=0):
    optimizer=torch.optim.Adam(model.parameters(), lr=lr,weight_decay=w2_lambda,amsgrad=True) 
    model=model.t0(device)
    model=model.train()  
    np.random.seed(seed)
    num_pat=len(pat_idx_list)
    pat_idx_list=np.array(pat_idx_list)
    assert max(pat_idx_list)<loader.pat_num
    for Epoch in range(epochs):
        np.random.shuffle(pat_idx_list)
        for chunk in range( int(num_pat/batch_size) ):
            batch_ids=pat_idx_list[chunk:chunk+batch_size]
            tmplist_11=[]
            tmplist_10=[]
            tmplist_target=[]
            for sample_id in batch_ids:
                seq11,seq10,target=loader.__getitem__(sample_id,gene_idx)
                 
                seq11=seq11.to(device)
                seq10=seq10.to(device)
                target=target.to(device)
                tmplist_11.append(seq11)
                tmplist_10.append(seq10)
                tmplist_target.append(target)
        tensor11=torch.cat(tmplist_11,dim=0)
        tensor10=torch.cat(tmplist_10,dim=0)
        tensor_target=torch.cat(tmplist_target,dim=0)
        
        if combine_mode==2:
           
            tensor_target=torch.cat(tmplist_target,dim=0)
            pred11=model(tensor11).squeeze(-1)
            pred10=model(tensor10).squeeze(-1)
            loss=MSELoss(reduction='mean')(0.5*( pred11+pred10),tensor_target)

        if combine_mode==1:
            pred_seqavg=model(0.5*(tensor11+tensor10)).squeeze(-1)
            loss=MSELoss(reduction='mean')(pred_seqavg,tensor_target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 

def eval_model(model,pat_idx_list,gene_idx,loader,verbose,device,save_dir ):
     
    model=model.eval()  
    model=model.to(device)
    pred_list_seqavg=[]
    pred_list_outavg=[]
    pred_list_seq11=[]
    pred_list_seq10=[]
    target_list=[]
    for sample_id in pat_idx_list:
        t0=time.time()
        seq11,seq10,target=loader.__getitem__(sample_id,gene_idx)
        seq11=seq11.to(device).unsqueeze(dim=0)
        seq10=seq10.to(device).unsqueeze(dim=0)
        pred_seqavg=model(0.5*(seq11+seq10)).squeeze(-1)
        pred_11=model(seq11).squeeze(-1)
        pred_10=model(seq10).squeeze(-1)
        pred_outavg=   0.5*( pred_11 +pred_10)  
        pred_list_seqavg.append(pred_seqavg.cpu().data.numpy()[0])
        pred_list_outavg.append(pred_outavg.cpu().data.numpy()[0])
        pred_list_seq11.append(pred_11.cpu().data.numpy()[0] )
        pred_list_seq10.append(pred_10.cpu().data.numpy()[0] )
        target_list.append(target.cpu().data.numpy())
        
        print('pat {} for gene {} done within {}s'.format(sample_id,gene_idx,time.time()-t0))
    p=pd.DataFrame()  
    p['expr_pred_11']=pred_list_seq11
    p['expr_pred_10']=pred_list_seq10
    p['expr_pred_seqavg']=pred_list_seqavg
    p['expr_pred_outavg']=pred_list_outavg
    p['expr_true']=target_list
    p.index=loader.exp_table_save.columns[pat_idx_list]
    p.to_csv('{}/{}.csv'.format(save_dir,gene_idx))
def init_model(name,left,right):
    from seq import testmlp
    mod_file='/scratch/deepnet/dna/savedmodel/{}.pt'.format(name)
    #device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu') #for local
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #for triton jobsubmission
    totallen=70000
    ARG_LEFT=left/totallen
    ARG_RIGHT=right/totallen
    params={'n_features':1+int(totallen*ARG_LEFT+totallen*ARG_RIGHT),'ker_size':[5]+[3]*6,'stride':[1]*5,\
                  'channels_list':[4]+[160,320,480,640,960],'dnn_list':[2,1],'di':[1,2,2,4,2] ,'res_on':1,\
                  'poolsize':[800,200,100,50,30],'ppoolstride':[200,8,2,2,2],'pooltype':3 }
    #above for 70019 2001.pt
    model=testmlp(**params)
    model.load_state_dict(torch.load(mod_file,map_location=device))
    model=model.to(device)
    return model

def eval_gene(tissue,modname,start_gene_id,end_gene_id,save_dir,debug,left,right):
    model=init_model(modname,left,right)
    totallen=70000
    loader=Patientwiseloader(tissue=tissue,chrom=list(range(1,23)),start=int(left/totallen),end=int(right/totallen))
    assert start_gene_id<len( loader.gene_ensemble_id )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
    print('{} genes in total '.format(len(loader.gene_ensemble_id)))
    gene_selected=loader.gene_ensemble_id[ start_gene_id: end_gene_id]
    print(gene_selected)
    prior=np.intersect1d( loader.sig_gene_id,gene_selected)
    rest=np.setdiff1d(  gene_selected,loader.sig_gene_id)
    import os
    for gene_id in np.concatenate((prior,rest)):
        if '{}.csv'.format(gene_id) in os.listdir(save_dir):
            continue
        t0=time.time()
        print('processing {}'.format(gene_id))
        eval_model(model,list(range(loader.pat_num)) if debug==0 else [0,1] ,gene_id,loader,0,device,save_dir  )
        print('done it takes {}s'.format(time.time()-t0))
        
        
def init_model_expecto( ):
    from seq import Beluga
    from xgboost import xgb
    #device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu') #for local
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #for triton jobsubmission
    xgbmod=None
    
    model=Beluga()
    model.load_state_dict(torch.load('/scratch/deepnet/yxz346/tmp/data/data/deepsea.beluga.pth'))
    model.eval()
    model.to(device)
    xgbmod=xgb.Booster()
    xgbmod.load_model('/scratch/deepnet/dna/Expecto_XgboostModel/expecto_Xgboost_opt.save')
    xgbmod.load_model('expecto_Xgboost_opt.save')
    return model,xgbmod

def eval_gene_expecto(tissue,modname,start_gene_id,end_gene_id,save_dir,debug):
    model=init_model(modname)
    loader=Patientwiseloader(tissue=tissue,chrom=list(range(1,23)))
    assert end_gene_id<len( loader.gene_ensemble_id )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
    print('{} genes in total '.format(len(loader.gene_ensemble_id)))
    for gene_id in loader.gene_ensemble_id[ start_gene_id: end_gene_id]:
        t0=time.time()
        print('processing {}'.format(gene_id))
        eval_model_expecto(model,list(range(loader.pat_num)) if debug==0 else [0,1] ,gene_id,loader,0,device,save_dir  )
        print('done it takes {}s'.format(time.time()-t0))

def eval_model_expecto(model,pat_idx_list,gene_idx,loader,verbose,device,save_dir):
     
    model=model.eval()  
    model=model.to(device)
    pred_list=[]
    target_list=[]
    for sample_id in pat_idx_list:
        t0=time.time()
        seq11,seq10,target=loader.__getitem__(sample_id,gene_idx)
        seq11=seq11.to(device).unsqueeze(dim=0)
        seq10=seq10.to(device).unsqueeze(dim=0)
        pred=model(0.5*(seq11+seq10)).squeeze(-1)
        pred_list.append(pred.cpu().data.numpy()[0])
         
        target_list.append(target.cpu().data.numpy())
        
        print('pat {} for gene {} done within {}s'.format(sample_id,gene_idx,time.time()-t0))
    p=pd.DataFrame()  
    p['expr_pred']=pred_list
    p['expr_true']=target_list
    p.index=loader.exp_table_save.columns[pat_idx_list]
    p.to_csv('{}/{}.csv'.format(save_dir,gene_idx))

def expecto_tensorinput_to_20020features(mod,datax,device):
    save_x=[]
    mid=int(datax.size()[-1] /2)
    for j in range(200):
        binx=datax[:,:,mid-200*100-900+j*200:mid-200*100+900+(j+1)*200]
         
        binx = binx.to(device)
        output=mod.forward(torch.unsqueeze(binx,2))#since mod uses 2d convolution,dimensions should be raised,from N,C,L to N,C,1,L
        avg_output= output 
        save_x.append(avg_output.cpu().data.numpy())
        mx=[]
        for ai in [0.01,0.02,0.05,0.1,0.2]:
            basepos=np.zeros(save_x[0].shape)
            baseneg=np.zeros(save_x[0].shape)
            for j in range(100):#as in the paper, For example, the −200bp to 0bp bin has a distance of −100 bp
                baseneg = baseneg + save_x[j]*np.exp(-ai*abs( j-99.5) )  
            mx.append(baseneg)
            for j in range(100,200):
                basepos = basepos + save_x[j]*np.exp(-ai*abs(j-99.5))  
            mx.append(basepos)
    return np.stack(mx)


#os.chdir('C:\\Users\\Administrator\\Desktop\\seqex\\pysrc\\config')
def regr():
    gene2snp_pickle_file= open('/scratch/deepnet/dna/allreplace_xin/gene2snp.pkl' ,'rb') 
    gene2snp=pickle.load(gene2snp_pickle_file)
    gene2snp_pickle_file.close()
    #eqtl_pat_pickle_file= open('gene4inf.pkl' ,'rb') 
    eqtl_pat_pickle_file= open('/scratch/deepnet/dna/allreplace_xin/gene4info.pkl' ,'rb') 
    eqtl_pat=pickle.load(eqtl_pat_pickle_file)
    eqtl_pat_pickle_file.close()
    geneinfo=pd.read_csv('gene_chr_tss_stra.csv').set_index('geneid')
    #gene_id='ENSG00000254699'
    for gene_id in gene2snp.keys():
        expr_df=pd.read_csv('{}.csv'.format(gene_id))
        expr_df['expr_pred']
        expr_df['expr_true']
        geneinfo.loc[gene_id]['chrnum']
        gene2snp[gene_id]





#test
import pickle
import pandas as pd
import numpy as np
import torch
import os
def generate_gene_chr_tss_stra():
    os.chdir('C:/Users/Administrator/Desktop/seqex/pysrc/config')
    pickle_file=open('C:/Users/Administrator/Desktop/seqex/pysrc/config/gene4inf.pkl','rb') 
    out=pickle.load(pickle_file)
    
    chrnum=[]
    genetssloc=[]
    genename=[]
    strand=[]
    
    for k in out.keys():
        genename.append( out[k][0] )
        chrnum.append( out[k][1][3:] )
        genetssloc.append( out[k][2] )
        strand.append( out[k][3] )
        
    gene_chr_tss_stra=pd.DataFrame()
    gene_chr_tss_stra['geneid']=genename
    gene_chr_tss_stra['chrnum']=chrnum
    gene_chr_tss_stra['tss']=genetssloc
    gene_chr_tss_stra['strand']=strand
    gene_chr_tss_stra.to_csv('gene_chr_tss_stra.csv',index=False)
    pickle_file.close()


def test_generate_gene_chr_tss_stra():
    a=pd.read_csv('gene_chr_tss_stra.csv',index_col=0)
    print(a)
    print(a.iloc[:3,:])
    print(a.loc['ENSG00000000457'])
#test_generate_gene_chr_tss_stra()


def get_chrsize(gene_annotation = "/scratch/deepnet/dna/VCF/geneanno.csv" if os.name!='nt' else \
    'C:/Users/Administrator/Desktop/seqex/pysrc/config/geneanno.csv',\
    hg19_fasta="/scratch/deepnet/dna/VCF/hg19.fa" if os.name!='nt' else \
    'C:/Users/Administrator/Desktop/seqex/pysrc/hg19/hg19.fa'):
    
    Genome = pyfasta.Fasta(hg19_fasta)
    TSS_strand = dict()
    
    for i, line in enumerate(open(gene_annotation)): 
        id, symbol, Chr, strand, TSS, CAGE_TSS, type = line.rstrip().split(",")
        if (i > 0): 
            if Chr not in TSS_strand: 
                TSS_strand[Chr] = []
            TSS_strand[Chr].append((id, Chr, int(CAGE_TSS), (1 if strand == "+" else -1)))
        TSS_strand = dict()

    for i, line in enumerate(open(gene_annotation)): 
        id, symbol, Chr, strand, TSS, CAGE_TSS, type = line.rstrip().split(",")
        if (i > 0): 
            if Chr not in TSS_strand: 
                TSS_strand[Chr] = []
            TSS_strand[Chr].append((id, Chr, int(CAGE_TSS), (1 if strand == "+" else -1)))
     
    chrname=[]
    chrlen=[]
    for Chr in TSS_strand.keys():
        chrname.append(Chr)
        chrlen.append( len(Genome[Chr]) )
    p=pd.DataFrame()
    p['chrname']=chrname
    p['chrlen']=chrlen
    p=p.set_index('chrname')
    p.to_csv('C:/Users/Administrator/Desktop/seqex/pysrc/config/chrsize.csv',index=True)
     
    
def get_chrsize_dataframe(file='C:/Users/Administrator/Desktop/seqex/pysrc/config/chrsize.csv'):
    get_chrsize()
     
    a=pd.read_csv(file).set_index('chrname')
    return a
    
def generate_freq(tiss='Ovary'):
    
    exp_folder='/scratch/deepnet/yxz346/tmp/data/data/RawExpression/'
    expressionfile=exp_folder+'chr{}/exp_raw/'.format(21)  +'{}/'.format(tiss)+'RAW.txt'  
    exp_table=pd.read_csv(expressionfile,index_col=0,sep='\t' )
    patientnames=exp_table.columns
    sample = pd.read_csv("/scratch/deepnet/dna/VCF/GTeXix.csv", header = None)
    tissue_id_in_vcf=[i+5 for i in range(450) if sample.iloc[i,0].split('-')[1] in [y.split('-')[1] for y in patientnames] ]
    snp_id=[]
    freq=[]
    for chr_num in range(1,23):
        
        a=pd.read_csv('/scratch/deepnet/dna/VCF/VCF/chr{}.vcf'.format(chr_num),sep='\t',header=None)
        for row_idx in range(a.shape[0]):
            for col_idx in tissue_id_in_vcf:
                count=0
                x=a.iloc[row_idx,col_idx]
                if x.split(':')[0] == '1/1'\
                     or (  x.split(':')[0] == './.' and \
                         np.argmax(np.array(x.split(':')[1].split(','),dtype=float))==2):
                    count=count+2
                if x.split(':')[0] == '0/1'\
                     or (  x.split(':')[0] == './.' and \
                         np.argmax(np.array(x.split(':')[1].split(','),dtype=float))==1): 

                    count=count+1
                else:
                    continue
            snp_id.append(a.iloc[row_idx,1])
            freq=count/(2*len(tissue_id_in_vcf))
            if row_idx%10000==9999:
                print('{}% for chro {} are done'.format(100*row_idx/a.shape[0],chr_num))
        print('chromo {} done'.format(chr_num))

    pout=pd.DataFrame()
    pout['variant_id']=snp_id
    pout['freq']=freq
    pout._csv('/scratch/deepnet/dna/VCF/tisfreq/{}_variant_id_freq_{}patients.csv'.format(tiss,len(tissue_id_in_vcf)),index=False)



def gene_unique_pd():
     
    pickle_mf=open('C:/Users/Administrator/Desktop/seqex/pysrc/config/UniqueSNPfilterByMAF_max.pkl','rb') 
    
    maf=pickle.load(pickle_mf)
    l=[]
    p=pd.DataFrame()
    for chromo in range(1,23):
        s=pd.DataFrame.from_dict( maf['chr{}'.format(chromo)],orient='index').values.ravel()
        l.append(s)
    p['0']=np.concatenate(l,axis=0)
    pickle_mf.close()
    p.to_csv('unique_eqtl_chrnumisacol.csv')

#test 
'''    
pickle_file= open('MAF4snp.pkl' ,'rb') 
f=pickle.load(pickle_file)
pickle_file.close()
vcf21=pd.read_csv('chr21.vcf',sep='\t',header=None)
loc_du=[]
for i in range(len(qtlnames)):
    if qtlnames[i].split('_')[1]==duplicates[0]:
        print(i)
duplicates = []
for item in nameslist:
    if nameslist.count(item) > 1:
        duplicates.append(item)
        break
print(duplicates)

sigpair=pd.read_csv('Ovary_Analysis.v6p.signif_snpgene_pairs.txt',sep='\t')
def get_snp_names(snplist):
    return [[x.split('_')[0]+'_'+x.split('_')[1]] for x in snplist],[x.split('_')[3] for x in snplist]


snploc,alt=get_snp_names(sigpair['variant_id'])

 


len(signames)-len(set(signames))
repeated=[]
repsnp=[]
contentlist=[]
for i in range(len(snploc)):
    if snploc.count(snploc[i])>1:
        
        content=[alt[j]  for j in range(len(snploc)) if snploc[i]==snploc[j]]
        if 1<len(set(content)):
            contentlist.append(content)
            repeated.append(i)
            repsnp.append(snploc[i])
            print('done')
sigpair.iloc[110:120,:]
sig=sigpair.set_index('variant_id')
sig.loc['1_1176597_T_C_b37']

sig.loc[sigpair['variant_id'].iloc[3918]]
sig.loc[sigpair['variant_id'].iloc[3877]]
'''