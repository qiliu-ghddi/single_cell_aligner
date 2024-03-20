import scanpy as sc
from tqdm import tqdm
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import numpy as np
import scipy.stats as stats
import re
import json
import pickle
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Single cell perturbation data preprocessing.')
    parser.add_argument('--adata_file', default='/home/zgzheng/Project/database_download/kaggle/data/h5ad_result/adata_train.h5ad', type=str, help='adata file')
    parser.add_argument('--save', default='/home/zqliu02/code/gene-drug-perturbation-model/data/kaggle_data_4celltype_all/perturb_processed.h5ad', type=str, help='save file dir')
    parser.add_argument('--conditionname', default='SMILES', type=str, help='save file dir')
    parser.add_argument('--ctrlname', default='C[S+](C)[O-]', type=str, help='save file dir')
    parser.add_argument('--gene_name', default=None, type=str, help='save file dir')
    parser.add_argument('--cell_type', default='cell_type', type=str, help='save file dir')
    parser.add_argument('--model_type', default='drug', type=str, help='model type')
    parser.add_argument('--cell_type_list', default=['T regulatory cells',  'NK cells', 'T cells CD8+', 'T cells CD4+'], help='save file dir')
    args = parser.parse_args()
    adata = sc.read_h5ad(args.adata_file)

 
    
    # adata.X = csr_matrix(adata.X.toarray())
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    adata.obs.loc[:, 'condition'] = adata.obs.apply(lambda x: "ctrl" if x[args.conditionname] == args.ctrlname else '+'.join([x[args.conditionname], "ctrl"]), axis=1)
    
    print(adata.obs["condition"])
    
    if args.gene_name == None:
        adata.var['gene_name'] = adata.var_names
    else:
        adata.var['gene_name'] = adata.var[args.gene_name]
    

    mt_genes = adata.var["gene_name"].str.startswith('MT-') 
    adata = adata[:, ~mt_genes]
    rb_genes = adata.var["gene_name"].str.startswith(("RPS", "RPL")) 
    adata = adata[:, ~rb_genes]
    
    adata.obs['cell_type'] = adata.obs[args.cell_type]

    
  

    if args.cell_type_list is not None:
        adata = adata[adata.obs["cell_type"].isin(args.cell_type_list)]
    
    adata.obs['cell_type'] = "NK and T cells"

    indices_to_keep = []


    for cell_type in adata.obs["cell_type"].unique():
      
        current_data = adata[adata.obs["cell_type"] == cell_type]
        
   
        condition_counts = current_data.obs["condition"].value_counts()
        
    
        conditions_to_keep = condition_counts[condition_counts >= 10].index
        
 
        indices_to_keep.extend(current_data.obs.index[current_data.obs["condition"].isin(conditions_to_keep)])

   
    adata = adata[adata.obs.index.isin(indices_to_keep)]



    if args.model_type == "gene":
        # 遍历每个基因名称并拆分
        split_gene_names = []
        for gene_name in adata.obs['condition']:
            # 使用 '+' 符号拆分基因名称
            parts = gene_name.split('+')
            
            # 去掉 'ctrl' 并添加到新列表中
            clean_parts = [part for part in parts if part != 'ctrl']
            
            # 将拆分后的部分添加到新列表
            split_gene_names.extend(clean_parts)

        missing_genes = [gene for gene in split_gene_names if gene not in adata.var_names]
        missing_genes = list(set(missing_genes))
        print("缺失的基因名:", set(missing_genes))
        pert_miss_gene_names = []
        for gene_name in adata.obs['condition']:
            # 使用 '+' 符号拆分基因名称
            parts = gene_name.split('+')
            
            # 去掉 'ctrl' 并添加到新列表中
            if parts==['ctrl']:
                continue
            if parts[0] in missing_genes or parts[1] in missing_genes:
                pert_miss_gene_names.append(gene_name)
        adata = adata[~adata.obs['condition'].isin(list(set(pert_miss_gene_names)))]

        sc.pp.highly_variable_genes(adata, n_top_genes=5000)
        print(adata)

        split_gene_names.extend(list(adata.var_names[adata.var['highly_variable']]))
        split_gene_names = list(set(split_gene_names))
        split_gene_names = [gene_name for gene_name in split_gene_names if gene_name not in pert_miss_gene_names]
        print(len(split_gene_names))
        print(adata.var['highly_variable'])
        print(adata.obs['condition'])
        # 创建一个空列表来存储拆分后的基因名称
        var_names = split_gene_names
        missing_genes = [gene_name for gene_name in split_gene_names if gene_name not in var_names]

        # 打印不存在于 AnnData 中的基因名称
        print("不存在于 AnnData 中的基因名称:", missing_genes)

        # 打印缺失的基因名

        # 选择高可变基因
        bool_index = np.isin(adata.var_names, split_gene_names)
        adata = adata[:,bool_index]
    elif args.model_type=="drug":
        sc.pp.highly_variable_genes(adata, n_top_genes=5000)
        adata = adata[:, adata.var["highly_variable"]]
  
    for col in adata.var.select_dtypes('category').columns:
        adata.var[col] = adata.var[col].astype(str)
    for col in adata.obs.select_dtypes('category').columns:
        adata.obs[col] = adata.obs[col].astype(str)
    adata.write(args.save)
    print(adata)
