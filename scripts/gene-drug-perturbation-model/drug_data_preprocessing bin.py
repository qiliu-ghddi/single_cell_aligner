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
from scgpt.preprocess import Preprocessor
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Single cell perturbation data preprocessing.')
    parser.add_argument('--adata_file', default='/home/lushi02/project/sl_data/LINCS/K562_compoud188.h5ad', type=str, help='adata file')
    parser.add_argument('--save', default='/home/zqliu02/code/gene-drug-perturbation-model/data/k562_compound188_1200_bin/perturb_processed.h5ad', type=str, help='save file dir')
    parser.add_argument('--conditionname', default='canonical_smiles', type=str, help='save file dir')
    parser.add_argument('--ctrlname', default='Vehicle', type=str, help='save file dir')
    parser.add_argument('--gene_name', default='symbol', type=str, help='save file dir')
    parser.add_argument('--cell_type', default='cell_type', type=str, help='save file dir')
    parser.add_argument('--n_hvg', default=1200, type=int, help='n_hvg')
    parser.add_argument('--n_bins', default= 51, type=int, help='n_bins')
    parser.add_argument('--data_is_raw', default=True, type=lambda x: (str(x).lower() == 'true'), help='data_is_raw')
    args = parser.parse_args()
    adata = sc.read_h5ad(args.adata_file)
 

    adata.obs.loc[:, 'condition'] = adata.obs.apply(lambda x: "ctrl" if x[args.conditionname] == args.ctrlname else '+'.join([x[args.conditionname], "ctrl"]), axis=1)
    
    print(adata.obs["condition"])
    

    if args.gene_name == None:
        adata.var['gene_name'] = adata.var_names
    else:
        adata.var['gene_name'] = adata.var[args.gene_name]
    
    adata.obs['cell_type'] = adata.obs[args.cell_type]

    preprocessor_perturb = Preprocessor(
    use_key="X",  # the key in adata.layers to use as raw data
    filter_gene_by_counts=3,  # step 1
    filter_cell_by_counts=False,  # step 2
    normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
    result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
    log1p=args.data_is_raw,  # 4. whether to log1p the normalized data
    result_log1p_key="X_log1p",
    subset_hvg=args.n_hvg,  # 5. whether to subset the raw data to highly variable genes
    hvg_flavor="seurat_v3" if args.data_is_raw else "cell_ranger",
    binning=args.n_bins,  # 6. whether to bin the raw data and to what number of bins
    result_binned_key="X_binned",  # the key in adata.layers to store the binned data
    )
    preprocessor_perturb(adata)
    
    adata.X = csr_matrix(adata.layers["X_binned"])
    
    for col in adata.var.select_dtypes('category').columns:
        adata.var[col] = adata.var[col].astype(str)
    for col in adata.obs.select_dtypes('category').columns:
        adata.obs[col] = adata.obs[col].astype(str)
    adata.write(args.save)
    print(adata)

