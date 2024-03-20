#!/bin/bash


# 运行Python脚本 with 参数
CUDA_VISIBLE_DEVICES=1 /home/cliang02/work/bin/cre-python Perturbation-GPU-pred.py \
    --load_model "save/dev_perturb_adamsonsldata-Mar14-10-51" \
    --eval_batch_size 64 \
    --use_fast_transformer true \
    --adata_file "/home/zqliu02/code/scGPT-GO-cell_embedding/data/adamsonsldata/perturb_processed.h5ad" \
    --data_name "adamsonsldata" \
    --model_type "gene"
  


# # 运行Python脚本 with 参数
# CUDA_VISIBLE_DEVICES=1 /home/cliang02/work/bin/cre-python Perturbation-GPU-pred.py \
#     --load_model "save/dev_perturb_kaggle_data_4celltype-Jan22-23-34-best-model" \
#     --eval_batch_size 64 \
#     --use_fast_transformer true \
#     --adata_file "/home/zqliu02/code/gene-drug-perturbation-model/data/kaggle_data_4celltype/perturb_processed.h5ad" \
#     --data_name "kaggle_data_4celltype" \
#     --model_type "drug"
  



