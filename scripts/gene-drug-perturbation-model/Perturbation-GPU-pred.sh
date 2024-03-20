#!/bin/bash

CUDA_VISIBLE_DEVICES=1 /home/cliang02/work/bin/cre-python Perturbation-GPU-pred.py \
    --load_model "/home/zqliu02/code/share/save/scGPT_human" \
    --batch_size 64 \
    --use_fast_transformer true \
    --adata_file "/home/zqliu02/code/scGPT-GO-cell_embedding/data/adamsonsldata/perturb_processed.h5ad" \
    --data_name "adamsonsldata" \
    --model_type "gene"
  



