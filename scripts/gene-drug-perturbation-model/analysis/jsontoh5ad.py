import scanpy as sc
import anndata as ad
from tqdm import tqdm
import os
import json
from anndata import concat
import pandas as pd
import pickle
import numpy as np
import glob
import argparse
def jsontoctrlh5ad(gene1,cell_type):


    safe = gene1.replace("\\", "_").replace("/", "_").replace("*", "_")
    if  not os.path.exists(f"{args.savedir}/{cell_type}_{safe}.json"):
        return None
    with open(f"{args.savedir}/{cell_type}_{safe}.json", "r") as json_file:
        gene1pred = json.load(json_file)

    if  gene1.rpartition('+')[0]=="ctrl":
        pert = gene1.rpartition('+')[2]
        gene1pred = np.array(gene1pred[pert])
    elif  gene1.rpartition('+')[2]=="ctrl":
        pert = gene1.rpartition('+')[0]
        gene1pred = np.array(gene1pred[pert])
    else:
        pert = gene1.split("+")
        gene1pred = np.array(gene1pred["_".join(pert)])
    #GPT2
    if gene_number<gene1pred.shape[1]:
        gene1pred = gene1pred[:,1:-1]
    

    return gene1pred 
def jsontopredh5ad(gene1,cell_type):
    safe = gene1.replace("\\", "_").replace("/", "_").replace("*", "_")
    if  not os.path.exists(f"{args.savedir}/{cell_type}_{safe}.json"):
        return None
    with open(f"{args.savedir}/{cell_type}_{safe}.json", "r") as json_file:
        gene1pred = json.load(json_file)

    if  gene1.rpartition('+')[0]=="ctrl":
        pert = gene1.rpartition('+')[2]
        gene1pred = np.array(gene1pred[pert])
    elif  gene1.rpartition('+')[2]=="ctrl":
        pert = gene1.rpartition('+')[0]
        gene1pred = np.array(gene1pred[pert])
    else:
        pert = gene1.split("+")
        gene1pred = np.array(gene1pred["_".join(pert)])
    if gene_number<gene1pred.shape[1]:
        gene1pred = gene1pred[:,1:-1]
    
    return gene1pred 


if __name__ == "__main__":


    # settings for data prcocessing
    parser = argparse.ArgumentParser(description='json to h5ad.')
    parser.add_argument('--adata_file', default='/home/zqliu02/code/gene-drug-perturbation-model/data/kaggle_data_nkcells/perturb_processednew.h5ad', type=str, help='Path to adata file')
    parser.add_argument('--savedir', default='/home/zqliu02/code/gene-drug-perturbation-model/save/dev_perturb_kaggle_data_nkcells-Jan24-01-03', type=str, help='Path to result file')
    parser.add_argument('--splits', nargs='+', default="/home/zqliu02/code/gene-drug-perturbation-model/data/kaggle_data_nkcells/splits/kaggle_data_nkcells_simulation_1_0.75.pkl", help='test pertlist')
    args = parser.parse_args()
    with open(args.splits, 'rb') as file:
    # 读取文件内容
        splits_data = pickle.load(file)
    adata_truth = sc.read_h5ad(args.adata_file)

    gene_number = adata_truth.shape[1]
    

    adata = ad.AnnData()
    adatactrl = ad.AnnData()
    adata_list = []  # 用于存储每个Anndata对象
    ctrl_adata_list = []
    combined_list = splits_data["test"]
    for cell_type in set(adata_truth.obs["cell_type"]):
        for condition in tqdm(combined_list, desc="Processing conditions"):
            
            pred = jsontopredh5ad(condition,cell_type)
            ctrl = jsontoctrlh5ad(condition,cell_type)
            if pred is None or ctrl is None:
                continue
            # 创建新观察列
            new_column = [condition] * len(pred)
            ctrl_column = [condition+"(ctrl)"] * len(ctrl)

            # 创建新数据
            new_data = np.array(pred)
            ctrl_data = np.array(ctrl)

            # 创建包含新观察列的 DataFrame
            new_obs = pd.DataFrame({'condition': new_column})
            new_obs['cell_type'] = cell_type
            ctrl_obs = pd.DataFrame({'ctrl_condition': ctrl_column})

            # 创建一个包含新数据和新观察列的 Anndata 对象
            new_adata = ad.AnnData(X=new_data, obs=new_obs)
            ctrl_adata = ad.AnnData(X=ctrl_data, obs=ctrl_obs)

            adata_list.append(new_adata)  # 将每个Anndata对象添加到列表
            ctrl_adata_list.append(ctrl_adata)
    # 使用anndata.concat连接所有Anndata对象
    adata = concat(adata_list, join="outer", axis=0)
    adatactrl = concat(ctrl_adata_list, join="outer", axis=0)

        # 将Anndata对象保存为H5AD文件

    adata.var['gene_name']=list(adata_truth.var.index)
    adata.var['gene_name'] = adata.var['gene_name'].astype(str)

    adatactrl.var['gene_name']=list(adata_truth.var.index)
    adatactrl.var['gene_name'] = adata.var['gene_name'].astype(str)

    os.makedirs(args.savedir+'/h5ad', exist_ok=True)
    adata.write_h5ad(args.savedir+'/h5ad/scGPTpred.h5ad') # type: ignore
    adatactrl.write_h5ad(args.savedir+'/h5ad/scGPTpredctrl.h5ad') # type: ignore

    adatatrainpert = adata_truth[adata_truth.obs.condition.isin(splits_data["train"])] # type: ignore
    adatatestpert = adata_truth[adata_truth.obs.condition.isin(splits_data["test"])] # type: ignore
    adatactrl =  adata_truth[adata_truth.obs.condition=="ctrl"] # type: ignore

    adatatrainpert.write(args.savedir+'/h5ad/traintruth.h5ad') # type: ignore
    adatatestpert.write(args.savedir+'/h5ad/testtruth.h5ad') # type: ignore
    adatactrl.write(args.savedir+'/h5ad/ctrl.h5ad') # type: ignore

    

    # delete file
    patterns = [os.path.join(args.savedir, '*+ctrl.json'), 
            os.path.join(args.savedir, '*+ctrlreal.json'),
             os.path.join(args.savedir, '*real.json'),
             os.path.join(args.savedir, '*+*.json')]
    


    files_to_delete = []
    for pattern in patterns:
        files_to_delete.extend(glob.glob(pattern))

    # 遍历找到的文件并删除它们
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except OSError as e:
            print(f"Error: {file_path} : {e.strerror}")




