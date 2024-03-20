import scanpy as sc
from tqdm import tqdm
import scipy.sparse as sp
import numpy as np
import scipy.stats as stats
import re
import json
import pickle
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error as mse
import argparse
import os
import scipy
import anndata as ad

def direction(predmeans,ctrlmeans,realmeans,genede_idx):
    predmeans =predmeans[genede_idx]
    ctrlmeans = ctrlmeans[genede_idx]
    realmeans = realmeans[genede_idx]

    preddirection = np.sign(predmeans - ctrlmeans)

    realdirection = np.sign(realmeans -ctrlmeans)
    consistent_signs = (preddirection == realdirection)
    direction = np.count_nonzero(consistent_signs) / len(genede_idx)
    return 1-direction
def perturb_variation(data1, data2,geneidx):
    # 示例数据
    #计算每列特征在data_set1中的值是否在data_set2的上下四分位点范围内
    data1 = data1[:,geneidx]
    data2 = data2[:,geneidx]
    
    q1_data2, q3_data2 = np.percentile(data2, [25, 75], axis=0) # type: ignore
    
    within_range = np.logical_and(data1 >= q1_data2, data1 <= q3_data2)
  
    iqr_ratio = np.mean(within_range, axis=0)

    return np.mean(iqr_ratio)
def STD(data1, data2,geneidx):
    # 示例数据
    data1 = data1[:,geneidx]
    data2 = data2[:,geneidx]

    # 计算均值和标准差
    mean_data1 = np.mean(data1, axis=0)
    std_data2 = np.std(data2, axis=0)
    mean_data2 = np.mean(data2, axis=0)

    # 计算上下界
    upper_bound_data2 = mean_data2 + std_data2
    lower_bound_data2 = mean_data2 - std_data2

    # 判断均值是否在标准差范围内，计算满足条件的列的比例
    within_range = np.logical_and(mean_data1 >= lower_bound_data2, mean_data1 <= upper_bound_data2)
    proportion_within_range = np.sum(within_range) / len(mean_data1)

    return  proportion_within_range


def Zscore(predicted_expression, true_expression,geneidx):
    # 示例数据

    # 计算均值和标准差
    z_scores = (predicted_expression.mean(axis=0) - true_expression.mean(axis=0)) / true_expression.std(axis=0)

    # 选取前20个差异表达基因的 z 分数
    de_genes_z_scores = z_scores[geneidx]

    # 计算每个基因的 z 分数均值
    mean_z_scores = np.mean(de_genes_z_scores, axis=0)
  

    return   mean_z_scores
def PearsonSD(pred,realmeans,genede_idx):
    predsmeans  = np.mean(pred[:,genede_idx], axis=0)
    realmeans = realmeans[genede_idx]
    
    pear = np.corrcoef(realmeans,predsmeans)[0, 1]
    

        
   
    return pear
def MSESD(pred,realmeans,genede_idx):
    predsmeans  = np.mean(pred[:,genede_idx], axis=0)
    realmeans = realmeans[genede_idx]

  

    mse_value = mse(predsmeans,realmeans)
    


    return mse_value
def direction_eval(adata,control_pred,savedir,pertlist,screen):


    result={}

    for cell_type in set(adata.obs["cell_type"]):
        ctrlreal = adata[(adata.obs["cell_type"]==cell_type) & (adata.obs.condition == 'ctrl') ].X.toarray() # type: ignore
        ctrlmeans = np.mean(ctrlreal, axis=0)

        for i in tqdm(pertlist, desc="Processing"):
            metic={}
            gene_raw2id = dict(zip(adata.var.index.values, adata.var.gene_name.values))
            gene2idx = {x: it for it, x in enumerate(adata.var.gene_name)}
            if adata[(adata.obs.cell_type == cell_type) & (adata.obs.condition == i)].shape[0] == 0:
                continue
            condition_name = i.replace("\\", "_").replace("/", "_").replace("*", "_")
            genede_idx20 = [
                    gene2idx[gene_raw2id[i]]
                    for i in adata.uns['top_non_zero_de_20'][f'{cell_type}_{condition_name}_1+1']
                ]
            genede_idx50 = [
                    gene2idx[gene_raw2id[i]]
                    for i in adata.uns['top_non_zero_de_50'][f'{cell_type}_{condition_name}_1+1']
                ]
            genede_idx100 = [
                    gene2idx[gene_raw2id[i]]
                    for i in adata.uns['top_non_zero_de_100'][f'{cell_type}_{condition_name}_1+1']
                ]
            genede_idx200 = [
                    gene2idx[gene_raw2id[i]]
                    for i in adata.uns['top_non_zero_de_200'][f'{cell_type}_{condition_name}_1+1']
                ]
            genede_idxall = [
                    gene2idx[gene_raw2id[i]]
                    for i in adata.uns['top_non_zero_de_all'][f'{cell_type}_{condition_name}_1+1']
                ]

            gene1pred = control_pred[(control_pred.obs["cell_type"]==cell_type) & (control_pred.obs.condition == re.sub("\+ctrl", "", i))].X.toarray()

            real = adata[(adata.obs["cell_type"]==cell_type) & (adata.obs.condition == i) ].X.toarray() # type: ignore
            realmeans = np.mean(real, axis=0)
            predmeans = np.mean(gene1pred, axis=0)
            if  np.all(realmeans[genede_idx20] == 0):
                continue
            
            metic['directionall']=direction(predmeans, ctrlmeans, realmeans, list(range(len(ctrlmeans))))
            metic['directiondeall']=direction(predmeans, ctrlmeans, realmeans, genede_idxall)
            metic['direction20']=direction(predmeans, ctrlmeans, realmeans, genede_idx20)
            metic['direction50']=direction(predmeans, ctrlmeans, realmeans, genede_idx50)
            metic['direction100']=direction(predmeans, ctrlmeans, realmeans, genede_idx100)
            metic['direction200']=direction(predmeans, ctrlmeans, realmeans, genede_idx200)

            

            result[cell_type+"_"+i]= metic
    directionall = 0
    directiondeall = 0
    direction20 = 0
    direction50 = 0
    direction100 = 0
    direction200= 0

    directionall_value = []
    directiondeall_value = []
    direction20_value = []
    direction50_value = []
    direction100_value = []
    direction200_value = []

    for j in result.keys():
        directionall = directionall +result[j]['directionall']
        directionall_value.append(result[j]['directionall'])

        directiondeall = directiondeall +result[j]['directiondeall']
        directiondeall_value.append(result[j]['directiondeall'])

        direction20 = direction20 +result[j]['direction20']
        direction20_value.append(result[j]['direction20'])

        direction50 = direction50 +result[j]['direction50']
        direction50_value.append(result[j]['direction50'])

        direction100 = direction100 +result[j]['direction100']
        direction100_value.append(result[j]['direction100'])

        direction200 = direction200 +result[j]['direction200']
        direction200_value.append(result[j]['direction200'])

    result['all'] = {'directionall': directionall/len(result.keys()),'directiondeall': directiondeall/len(result.keys()),'direction20':direction20/len(result.keys()),'direction50':direction50/len(result.keys()),'direction100':direction100/len(result.keys()),'direction200':direction200/len(result.keys())}

    result['allSD'] = {'directionall': np.array(directionall_value).std(),'directiondeall': np.array(directiondeall_value).std(),'direction20':np.array(direction20_value).std(),'direction50':np.array(direction50_value).std(),'direction100':np.array(direction100_value).std(),'direction200':np.array(direction200_value).std()}

    with open(f"{savedir}/test_metrics{screen}_direction.json", "w") as f:
        json.dump(result, f)

def normMSE_eval(adata,control_pred,savedir,pertlist,screen):
    result={}

    for cell_type in set(adata.obs["cell_type"]):
        ctrlreal = adata[(adata.obs["cell_type"]==cell_type) & (adata.obs.condition == 'ctrl') ].X.toarray() # type: ignore
        ctrlmeans = np.mean(ctrlreal, axis=0)
        for i in tqdm(pertlist, desc="Processing"):
            metic={}
            gene_raw2id = dict(zip(adata.var.index.values, adata.var.gene_name.values))
            gene2idx = {x: it for it, x in enumerate(adata.var.gene_name)}
            if adata[(adata.obs.cell_type == cell_type) & (adata.obs.condition == i)].shape[0] == 0:
                continue
            condition_name = i.replace("\\", "_").replace("/", "_").replace("*", "_")
            genede_idx20 = [
                    gene2idx[gene_raw2id[i]]
                    for i in adata.uns['top_non_zero_de_20'][f'{cell_type}_{condition_name}_1+1']
                ]
            genede_idx50 = [
                    gene2idx[gene_raw2id[i]]
                    for i in adata.uns['top_non_zero_de_50'][f'{cell_type}_{condition_name}_1+1']
                ]
            genede_idx100 = [
                    gene2idx[gene_raw2id[i]]
                    for i in adata.uns['top_non_zero_de_100'][f'{cell_type}_{condition_name}_1+1']
                ]
            genede_idx200 = [
                    gene2idx[gene_raw2id[i]]
                    for i in adata.uns['top_non_zero_de_200'][f'{cell_type}_{condition_name}_1+1']
                ]
            genede_idxall = [
                    gene2idx[gene_raw2id[i]]
                    for i in adata.uns['top_non_zero_de_all'][f'{cell_type}_{condition_name}_1+1']
                ]
            gene1pred = control_pred[(control_pred.obs["cell_type"]==cell_type) & (control_pred.obs.condition == re.sub("\+ctrl", "", i))].X.toarray()

            real = adata[(adata.obs["cell_type"]==cell_type) & (adata.obs.condition == i) ].X.toarray() # type: ignore
            realmeans = np.mean(real, axis=0)
            predmeans = np.mean(gene1pred, axis=0)
            if  np.all(realmeans[genede_idx20] == 0):
                continue

            noperturb_mse = mse(realmeans,ctrlmeans)
            noperturb_mse20 = mse(realmeans[genede_idx20],ctrlmeans[genede_idx20])
            noperturb_mse50 = mse(realmeans[genede_idx50],ctrlmeans[genede_idx50])
            noperturb_mse100 = mse(realmeans[genede_idx100],ctrlmeans[genede_idx100])
            noperturb_mse200 = mse(realmeans[genede_idx200],ctrlmeans[genede_idx200])
            noperturb_mseall = mse(realmeans[genede_idxall],ctrlmeans[genede_idxall])

            pred_mse = mse(realmeans,predmeans)
            pred_mse20 = mse(realmeans[genede_idx20],predmeans[genede_idx20])
            pred_mse50 = mse(realmeans[genede_idx50],predmeans[genede_idx50])
            pred_mse100 = mse(realmeans[genede_idx100],predmeans[genede_idx100])
            pred_mse200 = mse(realmeans[genede_idx200],predmeans[genede_idx200])
            pred_mseall = mse(realmeans[genede_idxall],predmeans[genede_idxall])

            metic['noperturb_mse']=noperturb_mse
            metic['noperturb_mse20']=noperturb_mse20
            metic['noperturb_mse50']=noperturb_mse50
            metic['noperturb_mse100']=noperturb_mse100
            metic['noperturb_mse200']=noperturb_mse200
            metic['noperturb_mseall']=noperturb_mseall

            metic['pred_mse']=pred_mse
            metic['pred_mse20']=pred_mse20
            metic['pred_mse50']=pred_mse50
            metic['pred_mse100']=pred_mse100
            metic['pred_mse200']=pred_mse200
            metic['pred_mseall']=pred_mseall
            result[i] = metic
    normalize = np.mean([sub_dict.get('noperturb_mse') for sub_dict in result.values()])
    normalize20 = np.mean([sub_dict.get('noperturb_mse20') for sub_dict in result.values()])
    normalize50 = np.mean([sub_dict.get('noperturb_mse50') for sub_dict in result.values()])
    normalize100 = np.mean([sub_dict.get('noperturb_mse100') for sub_dict in result.values()])
    normalize200 = np.mean([sub_dict.get('noperturb_mse200') for sub_dict in result.values()])
    normalizeall = np.mean([sub_dict.get('noperturb_mseall') for sub_dict in result.values()])

    pred = [sub_dict.get('pred_mse') for sub_dict in result.values()]
    pred20 = [sub_dict.get('pred_mse20') for sub_dict in result.values()]
    pred50 = [sub_dict.get('pred_mse50') for sub_dict in result.values()]
    pred100 = [sub_dict.get('pred_mse100') for sub_dict in result.values()]
    pred200 = [sub_dict.get('pred_mse200') for sub_dict in result.values()]
    predall = [sub_dict.get('pred_mseall') for sub_dict in result.values()]
    



    normalizeMSE = {}
    normalizeMSE['all'] = (np.array(pred, dtype=float)/normalize).mean()
    normalizeMSE['20'] =  (np.array(pred20, dtype=float)/normalize20).mean()
    normalizeMSE['50'] =  (np.array(pred50, dtype=float)/normalize50).mean()
    normalizeMSE['100'] =  (np.array(pred100, dtype=float)/normalize100).mean()
    normalizeMSE['200'] =  (np.array(pred200, dtype=float)/normalize200).mean()
    normalizeMSE['deall'] =  (np.array(predall, dtype=float) /normalizeall).mean()

    normalizeMSE['allSD'] = (np.array(pred, dtype=float)/normalize).std()
    normalizeMSE['20SD'] =  (np.array(pred20, dtype=float)/normalize20).std()
    normalizeMSE['50SD'] =  (np.array(pred50, dtype=float)/normalize50).std()
    normalizeMSE['100SD'] =  (np.array(pred100, dtype=float)/normalize100).std()
    normalizeMSE['200SD'] =  (np.array(pred200, dtype=float)/normalize200).std()
    normalizeMSE['deallSD'] =  (np.array(predall, dtype=float) /normalizeall).std()


    with open(f"{savedir}/test_metrics{screen}_normalizeMSE.json", "w") as f:
        json.dump(normalizeMSE, f)

def perturbvariation_eval(adata,control_pred,savedir,pertlist,screen):
    result={}
    for cell_type in set(adata.obs["cell_type"]):

        for i in tqdm(pertlist, desc="Processing"):
            metic={}
            gene_raw2id = dict(zip(adata.var.index.values, adata.var.gene_name.values))
            gene2idx = {x: it for it, x in enumerate(adata.var.gene_name)}
            if adata[(adata.obs.cell_type == cell_type) & (adata.obs.condition == i)].shape[0] == 0:
                continue
            condition_name = i.replace("\\", "_").replace("/", "_").replace("*", "_")
            genede_idx20 = [
                    gene2idx[gene_raw2id[i]]
                    for i in adata.uns['top_non_zero_de_20'][f'{cell_type}_{condition_name}_1+1']
                ]
            genede_idx50 = [
                    gene2idx[gene_raw2id[i]]
                    for i in adata.uns['top_non_zero_de_50'][f'{cell_type}_{condition_name}_1+1']
                ]
            genede_idx100 = [
                    gene2idx[gene_raw2id[i]]
                    for i in adata.uns['top_non_zero_de_100'][f'{cell_type}_{condition_name}_1+1']
                ]
            genede_idx200 = [
                    gene2idx[gene_raw2id[i]]
                    for i in adata.uns['top_non_zero_de_200'][f'{cell_type}_{condition_name}_1+1']
                ]
            genede_idxall = [
                    gene2idx[gene_raw2id[i]]
                    for i in adata.uns['top_non_zero_de_all'][f'{cell_type}_{condition_name}_1+1']
                ]

    
            gene1pred = control_pred[(control_pred.obs["cell_type"]==cell_type) & (control_pred.obs.condition == re.sub("\+ctrl", "", i))].X.toarray()
            real = adata[(adata.obs["cell_type"]==cell_type) & (adata.obs.condition == i) ].X.toarray() # type: ignore
            realmeans = np.mean(real, axis=0)
    
            if  np.all(realmeans[genede_idx20] == 0):
                continue


            metic['pert_variationall']=perturb_variation(gene1pred, real, list(range(real.shape[1])))
            metic['pert_variationdeall']=perturb_variation(gene1pred, real, genede_idxall)
            metic['pert_variation20']=perturb_variation(gene1pred, real, genede_idx20)
            metic['pert_variation50']=perturb_variation(gene1pred, real,genede_idx50)
            metic['pert_variation100']=perturb_variation(gene1pred, real, genede_idx100)
            metic['pert_variation200']=perturb_variation(gene1pred, real, genede_idx200)

            

            result[i]= metic


    pert_variationall = 0
    pert_variationdeall = 0
    pert_variation20 = 0
    pert_variation50 = 0
    pert_variation100 = 0
    pert_variation200= 0

    pert_variationall_value = []
    pert_variationdeall_value = []
    pert_variation20_value = []
    pert_variation50_value = []
    pert_variation100_value = []
    pert_variation200_value = []

    for j in result.keys():
        pert_variationall = pert_variationall +result[j]['pert_variationall']
        pert_variationall_value.append(result[j]['pert_variationall'])

        pert_variationdeall = pert_variationdeall +result[j]['pert_variationdeall']
        pert_variationdeall_value.append(result[j]['pert_variationdeall'])

        pert_variation20 = pert_variation20 +result[j]['pert_variation20']
        pert_variation20_value.append(result[j]['pert_variation20'])

        pert_variation50 = pert_variation50 +result[j]['pert_variation50']
        pert_variation50_value.append(result[j]['pert_variation50'])

        pert_variation100 = pert_variation100 +result[j]['pert_variation100']
        pert_variation100_value.append(result[j]['pert_variation100'])

        pert_variation200 = pert_variation200 +result[j]['pert_variation200']
        pert_variation200_value.append(result[j]['pert_variation200'])

    result['all'] = {'pert_variationall': pert_variationall/len(result.keys()),'pert_variationdeall': pert_variationdeall/len(result.keys()),'pert_variation20':pert_variation20/len(result.keys()),'pert_variation50':pert_variation50/len(result.keys()),'pert_variation100':pert_variation100/len(result.keys()),'pert_variation200':pert_variation200/len(result.keys())}

    result['allSD'] = {'pert_variationall': np.array(pert_variationall_value).std(),'pert_variationdeall': np.array(pert_variationdeall_value).std(),'pert_variation20':np.array(pert_variation20_value).std(),'pert_variation50':np.array(pert_variation50_value).std(),'pert_variation100':np.array(pert_variation100_value).std(),'pert_variation200':np.array(pert_variation200_value).std()}

    with open(f"{savedir}/test_metrics{screen}_pertvariation.json", "w") as f:
        json.dump(result, f)


def STD_eval(adata,control_pred,savedir,pertlist,screen):
    result={}
  
    for cell_type in set(adata.obs["cell_type"]):

        for i in tqdm(pertlist, desc="Processing"):
            metic={}
            gene_raw2id = dict(zip(adata.var.index.values, adata.var.gene_name.values))
            gene2idx = {x: it for it, x in enumerate(adata.var.gene_name)}
            if adata[(adata.obs.cell_type == cell_type) & (adata.obs.condition == i)].shape[0] == 0:
                continue
            condition_name = i.replace("\\", "_").replace("/", "_").replace("*", "_")
            genede_idx20 = [
                    gene2idx[gene_raw2id[i]]
                    for i in adata.uns['top_non_zero_de_20'][f'{cell_type}_{condition_name}_1+1']
                ]
            genede_idx50 = [
                    gene2idx[gene_raw2id[i]]
                    for i in adata.uns['top_non_zero_de_50'][f'{cell_type}_{condition_name}_1+1']
                ]
            genede_idx100 = [
                    gene2idx[gene_raw2id[i]]
                    for i in adata.uns['top_non_zero_de_100'][f'{cell_type}_{condition_name}_1+1']
                ]
            genede_idx200 = [
                    gene2idx[gene_raw2id[i]]
                    for i in adata.uns['top_non_zero_de_200'][f'{cell_type}_{condition_name}_1+1']
                ]
            genede_idxall = [
                    gene2idx[gene_raw2id[i]]
                    for i in adata.uns['top_non_zero_de_all'][f'{cell_type}_{condition_name}_1+1']
                ]

     

            gene1pred = control_pred[(control_pred.obs["cell_type"]==cell_type) & (control_pred.obs.condition == re.sub("\+ctrl", "", i))].X.toarray()

            real = adata[(adata.obs["cell_type"]==cell_type) & (adata.obs.condition == i) ].X.toarray() # type: ignore
            realmeans = np.mean(real, axis=0)
            if  np.all(realmeans[genede_idx20] == 0):
                continue
        
            metic['STDall']=STD(gene1pred, real, list(range(real.shape[1])))
            metic['STDdeall']=STD(gene1pred, real, genede_idxall)
            metic['STD20']=STD(gene1pred, real, genede_idx20)
            metic['STD50']=STD(gene1pred, real,genede_idx50)
            metic['STD100']=STD(gene1pred, real, genede_idx100)
            metic['STD200']=STD(gene1pred, real, genede_idx200)

            

            result[i]= metic

    STDall = 0
    STDdeall = 0
    STD20 = 0
    STD50 = 0
    STD100 = 0
    STD200= 0

    STDall_value = []
    STDdeall_value = []
    STD20_value = []
    STD50_value = []
    STD100_value = []
    STD200_value = []

    for j in result.keys():
        STDall = STDall +result[j]['STDall']
        STDall_value.append(result[j]['STDall'])

        STDdeall = STDdeall +result[j]['STDdeall']
        STDdeall_value.append(result[j]['STDdeall'])

        STD20 = STD20 +result[j]['STD20']
        STD20_value.append(result[j]['STD20'])

        STD50 = STD50 +result[j]['STD50']
        STD50_value.append(result[j]['STD50'])

        STD100 = STD100 +result[j]['STD100']
        STD100_value.append(result[j]['STD100'])

        STD200 = STD200 +result[j]['STD200']
        STD200_value.append(result[j]['STD200'])

    result['all'] = {'STDall': STDall/len(result.keys()),'STDdeall': STDdeall/len(result.keys()),'STD20':STD20/len(result.keys()),'STD50':STD50/len(result.keys()),'STD100':STD100/len(result.keys()),'STD200':STD200/len(result.keys())}

    result['allSD'] = {'STDall': np.array(STDall_value).std(),'STDdeall': np.array(STDdeall_value).std(),'STD20':np.array(STD20_value).std(),'STD50':np.array(STD50_value).std(),'STD100':np.array(STD100_value).std(),'STD200':np.array(STD200_value).std()}

    with open(f"{savedir}/test_metrics{screen}_STD.json", "w") as f:
        json.dump(result, f)
    #/home/cliang02/work/bin/cre-python /home/zqliu02/code/h5ad/评价scGPT/main-STD.py

        
        
def Zscore_eval(adata,control_pred,savedir,pertlist,screen):

    result={}
    for cell_type in set(adata.obs["cell_type"]):
        
        for i in tqdm(pertlist, desc="Processing"):
            metic={}
            gene_raw2id = dict(zip(adata.var.index.values, adata.var.gene_name.values))
            gene2idx = {x: it for it, x in enumerate(adata.var.gene_name)}
            if adata[(adata.obs.cell_type == cell_type) & (adata.obs.condition == i)].shape[0] == 0:
                continue
            condition_name = i.replace("\\", "_").replace("/", "_").replace("*", "_")
            genede_idx20 = [
                    gene2idx[gene_raw2id[i]]
                    for i in adata.uns['top_non_zero_de_20'][f'{cell_type}_{condition_name}_1+1']
                ]
            genede_idx50 = [
                    gene2idx[gene_raw2id[i]]
                    for i in adata.uns['top_non_zero_de_50'][f'{cell_type}_{condition_name}_1+1']
                ]
            genede_idx100 = [
                    gene2idx[gene_raw2id[i]]
                    for i in adata.uns['top_non_zero_de_100'][f'{cell_type}_{condition_name}_1+1']
                ]
            genede_idx200 = [
                    gene2idx[gene_raw2id[i]]
                    for i in adata.uns['top_non_zero_de_200'][f'{cell_type}_{condition_name}_1+1']
                ]
            genede_idxall = [
                    gene2idx[gene_raw2id[i]]
                    for i in adata.uns['top_non_zero_de_all'][f'{cell_type}_{condition_name}_1+1']
                ]

            gene1pred = control_pred[(control_pred.obs["cell_type"]==cell_type) & (control_pred.obs.condition == re.sub("\+ctrl", "", i))].X.toarray()
            real = adata[(adata.obs["cell_type"]==cell_type) & (adata.obs.condition == i) ].X.toarray() # type: ignore
            realmeans = np.mean(real, axis=0)
    
            if  np.all(realmeans[genede_idx20] == 0):
                continue
        
            metic['Zscoreall']=Zscore(gene1pred, real, list(range(real.shape[1])))
            metic['Zscoredeall']=Zscore(gene1pred, real, genede_idxall)
            metic['Zscore20']=Zscore(gene1pred, real, genede_idx20)
            metic['Zscore50']=Zscore(gene1pred, real,genede_idx50)
            metic['Zscore100']=Zscore(gene1pred, real, genede_idx100)
            metic['Zscore200']=Zscore(gene1pred, real, genede_idx200)

            

            result[i]= metic

    Zscoreall = 0
    Zscoredeall = 0
    Zscore20 = 0
    Zscore50 = 0
    Zscore100 = 0
    Zscore200= 0

    Zscoreall_value = []
    Zscoredeall_value = []
    Zscore20_value = []
    Zscore50_value = []
    Zscore100_value = []
    Zscore200_value = []

    for j in result.keys():
        Zscoreall = Zscoreall +result[j]['Zscoreall']
        Zscoreall_value.append(result[j]['Zscoreall'])

        Zscoredeall = Zscoredeall +result[j]['Zscoredeall']
        Zscoredeall_value.append(result[j]['Zscoredeall'])

        Zscore20 = Zscore20 +result[j]['Zscore20']
        Zscore20_value.append(result[j]['Zscore20'])

        Zscore50 = Zscore50 +result[j]['Zscore50']
        Zscore50_value.append(result[j]['Zscore50'])

        Zscore100 = Zscore100 +result[j]['Zscore100']
        Zscore100_value.append(result[j]['Zscore100'])

        Zscore200 = Zscore200 +result[j]['Zscore200']
        Zscore200_value.append(result[j]['Zscore200'])

    result['all'] = {'Zscoreall': Zscoreall/len(result.keys()),'Zscoredeall': Zscoredeall/len(result.keys()),'Zscore20':Zscore20/len(result.keys()),'Zscore50':Zscore50/len(result.keys()),'Zscore100':Zscore100/len(result.keys()),'Zscore200':Zscore200/len(result.keys())}

    result['allSD'] = {'Zscoreall': np.array(Zscoreall_value).std(),'Zscoredeall': np.array(Zscoredeall_value).std(),'Zscore20':np.array(Zscore20_value).std(),'Zscore50':np.array(Zscore50_value).std(),'Zscore100':np.array(Zscore100_value).std(),'Zscore200':np.array(Zscore200_value).std()}

    with open(f"{savedir}/test_metrics{screen}_Zscore.json", "w") as f:
        json.dump(result, f)
    #/home/cliang02/work/bin/cre-python /home/zqliu02/code/h5ad/评价scGPT/main-Zscore.py


def pearson_eval(adata,control_pred,savedir,pertlist,screen):
    
    result={}
    for cell_type in set(adata.obs["cell_type"]):
        ctrlreal = adata[(adata.obs["cell_type"]==cell_type) & (adata.obs.condition == 'ctrl') ].X.toarray() # type: ignore
        ctrlmeans = np.mean(ctrlreal, axis=0)
        for i in tqdm(pertlist, desc="Processing"):
            metic={}
            gene_raw2id = dict(zip(adata.var.index.values, adata.var.gene_name.values))
            gene2idx = {x: it for it, x in enumerate(adata.var.gene_name)}
            
            if adata[(adata.obs.cell_type == cell_type) & (adata.obs.condition == i)].shape[0] == 0:
                continue
           
            condition_name = i.replace("\\", "_").replace("/", "_").replace("*", "_")

            genede_idx20 = [
                    gene2idx[gene_raw2id[i]]
                    for i in adata.uns['top_non_zero_de_20'][f'{cell_type}_{condition_name}_1+1']
                ]
            genede_idx50 = [
                    gene2idx[gene_raw2id[i]]
                    for i in adata.uns['top_non_zero_de_50'][f'{cell_type}_{condition_name}_1+1']
                ]
            genede_idx100 = [
                    gene2idx[gene_raw2id[i]]
                    for i in adata.uns['top_non_zero_de_100'][f'{cell_type}_{condition_name}_1+1']
                ]
            genede_idx200 = [
                    gene2idx[gene_raw2id[i]]
                    for i in adata.uns['top_non_zero_de_200'][f'{cell_type}_{condition_name}_1+1']
                ]
            genede_idxall = [
                    gene2idx[gene_raw2id[i]]
                    for i in adata.uns['top_non_zero_de_all'][f'{cell_type}_{condition_name}_1+1']
                ]
            gene1pred = control_pred[(control_pred.obs["cell_type"]==cell_type) & (control_pred.obs.condition == re.sub("\+ctrl", "", i))].X.toarray()

            real = adata[(adata.obs["cell_type"]==cell_type) & (adata.obs.condition == i) ].X.toarray() # type: ignore

            realmeans = np.mean(real, axis=0)
            
            if  np.all(realmeans[genede_idx20] == 0):
                continue
            metic['Pearsonall'] =PearsonSD(gene1pred, realmeans, list(range(len(ctrlmeans))))
            metic['Pearsondeall'] =PearsonSD(gene1pred, realmeans, genede_idxall)
            metic['Pearsonde20']=PearsonSD(gene1pred, realmeans, genede_idx20)
            metic['Pearsonde50'] =PearsonSD(gene1pred, realmeans, genede_idx50)
            metic['Pearsonde100'] =PearsonSD(gene1pred, realmeans, genede_idx100)
            metic['Pearsonde200'] =PearsonSD(gene1pred, realmeans, genede_idx200)

            metic['Pearsonall_delate'] =PearsonSD(gene1pred-ctrlmeans, realmeans-ctrlmeans, list(range(len(ctrlmeans))))
            metic['Pearsondeall_delate'] =PearsonSD(gene1pred-ctrlmeans, realmeans-ctrlmeans, genede_idxall)
            metic['Pearsonde20_delate']=PearsonSD(gene1pred-ctrlmeans, realmeans-ctrlmeans, genede_idx20)
            metic['Pearsonde50_delate'] =PearsonSD(gene1pred-ctrlmeans, realmeans-ctrlmeans, genede_idx50)
            metic['Pearsonde100_delate'] =PearsonSD(gene1pred-ctrlmeans, realmeans-ctrlmeans, genede_idx100)
            metic['Pearsonde200_delate'] =PearsonSD(gene1pred-ctrlmeans, realmeans-ctrlmeans, genede_idx200)



            

            result[cell_type+"_"+i]= metic
    Pearsonall = []
    Pearsondeall = []
    Pearson20 = []
    Pearson50 = []
    Pearson100 = []
    Pearson200= []



    Pearsonall_delate = []
    Pearsondeall_delate = []
    Pearson20_delate = []
    Pearson50_delate = []
    Pearson100_delate = []
    Pearson200_delate= []



    for j in result.keys():
        Pearsonall.append(result[j]['Pearsonall'])
        Pearsondeall.append(result[j]['Pearsondeall'])
        Pearson20.append(result[j]['Pearsonde20'])
        Pearson50.append(result[j]['Pearsonde50'])
        Pearson100.append(result[j]['Pearsonde100'])
        Pearson200.append(result[j]['Pearsonde200'])



        Pearsonall_delate.append(result[j]['Pearsonall_delate'])
        Pearsondeall_delate.append(result[j]['Pearsondeall_delate'])
        Pearson20_delate.append(result[j]['Pearsonde20_delate'])
        Pearson50_delate.append(result[j]['Pearsonde50_delate'])
        Pearson100_delate.append(result[j]['Pearsonde100_delate'])
        Pearson200_delate.append(result[j]['Pearsonde200_delate'])


    result['all'] = {'Pearsonall': np.array(Pearsonall).mean(),'Pearsondeall': np.array(Pearsondeall).mean(),'Pearson20':np.array(Pearson20).mean(),'Pearson50':np.array(Pearson50).mean(),
                    'Pearson100':np.array(Pearson100).mean(),'Pearson200':np.array(Pearson200).mean(),
                    'Pearsonall_SD': np.array(Pearsonall).std(),'Pearsondeall_SD':  np.array(Pearsondeall).std(),'Pearson20_SD': np.array(Pearson20).std(),
                    'Pearson50_SD':np.array(Pearson50).std(),'Pearson100_SD':np.array(Pearson100).std(),'Pearson200_SD':np.array(Pearson200).std(),
                    'Pearsonall_delate': np.array(Pearsonall_delate).mean(),'Pearsondeall_delate': np.array(Pearsondeall_delate).mean(),'Pearson20_delate':np.array(Pearson20_delate).mean(),
                    'Pearson50_delate':np.array(Pearson50_delate).mean(),'Pearson100_delate':np.array(Pearson100_delate).mean(),'Pearson200_delate':np.array(Pearson200_delate).mean(),
                    'Pearsonall_SD_delate': np.array(Pearsonall_delate).std(),'Pearsondeall_SD_delate': np.array(Pearsondeall_delate).std(),
                    'Pearson20_SD_delate':np.array(Pearson20_delate).std(),'Pearson50_SD_delate':np.array(Pearson50_delate).std(),'Pearson100_SD_delate':np.array(Pearson100_delate).std(),
                    'Pearson200_SD_delate':np.array(Pearson200_delate).std()}

    with open(f"{savedir}/test_metrics{screen}_Pearson.json", "w") as f:
        json.dump(result, f)

def MSE_eval(adata,control_pred,savedir,pertlist,screen):
    result={}


    for cell_type in set(adata.obs["cell_type"]):
        ctrlreal = adata[(adata.obs["cell_type"]==cell_type) & (adata.obs.condition == 'ctrl') ].X.toarray() # type: ignore
        ctrlmeans = np.mean(ctrlreal, axis=0)

        for i in tqdm(pertlist, desc="Processing"):
            metic={}
            gene_raw2id = dict(zip(adata.var.index.values, adata.var.gene_name.values))
            gene2idx = {x: it for it, x in enumerate(adata.var.gene_name)}
            if adata[(adata.obs.cell_type == cell_type) & (adata.obs.condition == i)].shape[0] == 0:
                continue
            condition_name = i.replace("\\", "_").replace("/", "_").replace("*", "_")
            genede_idx20 = [
                    gene2idx[gene_raw2id[i]]
                    for i in adata.uns['top_non_zero_de_20'][f'{cell_type}_{condition_name}_1+1']
                ]
            genede_idx50 = [
                    gene2idx[gene_raw2id[i]]
                    for i in adata.uns['top_non_zero_de_50'][f'{cell_type}_{condition_name}_1+1']
                ]
            genede_idx100 = [
                    gene2idx[gene_raw2id[i]]
                    for i in adata.uns['top_non_zero_de_100'][f'{cell_type}_{condition_name}_1+1']
                ]
            genede_idx200 = [
                    gene2idx[gene_raw2id[i]]
                    for i in adata.uns['top_non_zero_de_200'][f'{cell_type}_{condition_name}_1+1']
                ]
            genede_idxall = [
                    gene2idx[gene_raw2id[i]]
                    for i in adata.uns['top_non_zero_de_all'][f'{cell_type}_{condition_name}_1+1']
                ]

            gene1pred = control_pred[(control_pred.obs["cell_type"]==cell_type) & (control_pred.obs.condition == re.sub("\+ctrl", "", i))].X.toarray()

            real = adata[(adata.obs["cell_type"]==cell_type) & (adata.obs.condition == i) ].X.toarray() # type: ignore

            realmeans = np.mean(real, axis=0)
            if  np.all(realmeans[genede_idx20] == 0):
                continue
        
         
            metic['MSEall']=MSESD(gene1pred, realmeans, list(range(len(ctrlmeans))))
            metic['MSEdeall'] =MSESD(gene1pred, realmeans, genede_idxall)
            metic['MSE20'] =MSESD(gene1pred, realmeans, genede_idx20)
            metic['MSE50'] =MSESD(gene1pred, realmeans, genede_idx50)
            metic['MSE100'] =MSESD(gene1pred, realmeans, genede_idx100)
            metic['MSE200'] =MSESD(gene1pred, realmeans, genede_idx200)



            

            result[cell_type+"_"+i]= metic
    MSEall = 0
    MSEdeall = 0
    MSE20 = 0
    MSE50 = 0
    MSE100 = 0
    MSE200= 0

    MSEall_value = []
    MSEdeall_value = []
    MSE20_value = []
    MSE50_value = []
    MSE100_value = []
    MSE200_value = []

    for j in result.keys():
        MSEall = MSEall +result[j]['MSEall']
        MSEall_value.append(result[j]['MSEall'])

        MSEdeall = MSEdeall +result[j]['MSEdeall']
        MSEdeall_value.append(result[j]['MSEdeall'])

        MSE20 = MSE20 +result[j]['MSE20']
        MSE20_value.append(result[j]['MSE20'])

        MSE50 = MSE50 +result[j]['MSE50']
        MSE50_value.append(result[j]['MSE50'])

        MSE100 = MSE100 +result[j]['MSE100']
        MSE100_value.append(result[j]['MSE100'])

        MSE200 = MSE200 +result[j]['MSE200']
        MSE200_value.append(result[j]['MSE200'])

    result['all'] = {'MSEall': MSEall/len(result.keys()),'MSEdeall': MSEdeall/len(result.keys()),'MSE20':MSE20/len(result.keys()),'MSE50':MSE50/len(result.keys()),'MSE100':MSE100/len(result.keys()),'MSE200':MSE200/len(result.keys())}

    result['allSD'] = {'MSEall': np.array(MSEall_value).std(),'MSEdeall': np.array(MSEdeall_value).std(),'MSE20':np.array(MSE20_value).std(),'MSE50':np.array(MSE50_value).std(),'MSE100':np.array(MSE100_value).std(),'MSE200':np.array(MSE200_value).std()}

    with open(f"{savedir}/test_metrics{screen}_MSE.json", "w") as f:
        json.dump(result, f)


if __name__ == "__main__":

    
    parser = argparse.ArgumentParser(description='evaluate the model.')
    parser.add_argument('--input_h5ad', default='/home/zqliu02/code/gene-drug-perturbation-model/save/dev_perturb_kaggle_data_4celltype-Jan23-18-05-NKcells split/h5ad', type=str, help='Path to h5ad file')
    parser.add_argument('--adata_file', default='/home/zqliu02/code/gene-drug-perturbation-model/data/kaggle_data_4celltype/perturb_processednew.h5ad', type=str, help='Path to adata file')
    parser.add_argument('--savedir', default='/home/zqliu02/code/gene-drug-perturbation-model/save/dev_perturb_kaggle_data_4celltype-Jan23-18-05-NKcells split', type=str, help='Path to result file')
    parser.add_argument('--screen', default='', type=str, help='screen')
    parser.add_argument('--splits', nargs='+', default="/home/zqliu02/code/gene-drug-perturbation-model/data/kaggle_data_4celltype/splits/kaggle_data_4celltype_simulation_1_0.75.pkl", help='splits file')
    parser.add_argument('--test_list', nargs='+', default=None, help='test list')
    parser.add_argument('--cell_type', nargs='+', default=None, help='cell type')  # "T regulatory cells" "NK cells" "T cells CD4+" "T cells CD8+"
    args = parser.parse_args()

    adata = sc.read_h5ad(args.adata_file)
  
    if args.splits is not None:
        with open(args.splits, 'rb') as file:
        # 读取文件内容
            splits_data = pickle.load(file)
        test_list = splits_data["test"]
    else:
        test_list = args.test_list
    
    input_h5ad_control = args.input_h5ad + "/ctrl.h5ad"
    input_h5ad_pred = args.input_h5ad + "/scGPTpred.h5ad"

    # Load input h5ad files and modify them.
    control = sc.read_h5ad(input_h5ad_control)
    control.obs_names_make_unique()
    control.obs_names = ["_".join(["cell", x]) for i,x in enumerate(control.obs_names)]
    control.obs["condition"] = "ctrl"
    # control.var_names = control.var.gene_name
    control.X = scipy.sparse.csr_matrix(control.X)
    
    pred = sc.read_h5ad(input_h5ad_pred)
    pred.obs_names_make_unique()
    pred.obs_names = ["_".join(["cell", x]) for i,x in enumerate(pred.obs_names)]
    pred.obs.condition = [re.sub("\+ctrl", "", x) for i,x in enumerate(pred.obs.condition)]
    pred.var_names = pred.var.gene_name
    pred.X = scipy.sparse.csr_matrix(pred.X)
    	
    control_pred = ad.concat([control, pred], index_unique = "-")

    if args.cell_type is not None:
        control_pred = control_pred[control_pred.obs["cell_type"] == args.cell_type]
        adata = adata[adata.obs["cell_type"] == args.cell_type]
        args.savedir = args.savedir + "/"+args.cell_type
        os.makedirs(args.savedir, exist_ok=True)
   
    pearson_eval(adata,control_pred,args.savedir,test_list,args.screen)
    
    MSE_eval(adata,control_pred,args.savedir,test_list,args.screen)
    direction_eval(adata,control_pred,args.savedir,test_list,args.screen)
    perturbvariation_eval(adata,control_pred,args.savedir,test_list,args.screen)
    normMSE_eval(adata,control_pred,args.savedir,test_list,args.screen)
    STD_eval(adata,control_pred,args.savedir,test_list,args.screen)
    Zscore_eval(adata,control_pred,args.savedir,test_list,args.screen)
