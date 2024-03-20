import json
import os
import sys
import time
import copy
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Union, Optional
import warnings
import scanpy as sc
import torch
import numpy as np
import matplotlib
from scipy.stats import pearsonr, spearmanr
from torch import nn
from torch.nn import functional as F
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
from torch_geometric.loader import DataLoader
from gears import PertData, GEARS
from gears.inference import compute_metrics, deeper_analysis, non_dropout_analysis
from gears.utils import create_cell_graph_dataset_for_prediction,get_similarity_network, GeneSimNetwork,create_drug_cell_graph_dataset_for_prediction
import argparse
sys.path.insert(0, "../")

import scgpt as scg
from scgpt.model import TransformerGenerator,TransformerModel
from scgpt.loss import (
    masked_mse_loss,
    criterion_neg_log_bernoulli,
    masked_relative_error,
    pertmean_mse_loss,
    weighted_mse_loss,
)
import subprocess
from scgpt.tokenizer import tokenize_batch, pad_batch, tokenize_and_pad_batch
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import set_seed, map_raw_id_to_vocab_id

matplotlib.rcParams["savefig.transparent"] = False
warnings.filterwarnings("ignore")

def get_gpu_memory_usage():
    # 执行 nvidia-smi 命令获取显存使用信息
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
    memory_used = result.stdout.decode('utf-8')
    
    # 提取显存使用量
    memory_used = [int(x) for x in memory_used.split('\n') if x.strip()]
    return memory_used

def train(model: nn.Module, train_loader: torch.utils.data.DataLoader) -> None:
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss, total_mse = 0.0, 0.0
    start_time = time.time()

    num_batches = len(train_loader)
    for batch, batch_data in enumerate(train_loader):
        batch_size = len(batch_data.pert)
        batch_data.to(device)
        x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 2)
        x= x.view(-1)
        ori_gene_values =  x.view(batch_size, -1)
        # ori_gene_values = ctrl_expression.repeat(batch_size, 1).to(dtype=torch.float32).to(device)

        target_gene_values = batch_data.y.view(batch_size, n_genes) # (batch_size, n_genes)
        if args.include_zero_gene in ["all", "batch-wise","highly_variable"]:
            if args.include_zero_gene == "all":
                input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
                if len(input_gene_ids) > args.max_seq_len:
                    input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[:args.max_seq_len]      
            if args.include_zero_gene == "batch-wise":
                input_gene_ids = (
                    ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                )
                
                if len(input_gene_ids) > args.max_seq_len:
                    input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[
                    :args.max_seq_len
                ]
            if args.include_zero_gene == "highly_variable":
                input_gene_ids = pert_data.highly_variable_gene_indices
                input_gene_ids = torch.tensor(input_gene_ids, device=device)
            input_values = ori_gene_values[:, input_gene_ids]
            
            target_values = target_gene_values[:, input_gene_ids]

            mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
            mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)
            src_key_padding_mask = torch.zeros_like(
                input_values, dtype=torch.bool, device=device
            )
        if args.model_type =="gene":
            pert_flags = batch_data.pert_flags
            pert_flags = torch.tensor(pert_flags).long().view(batch_size, n_genes)
            pert_flags = pert_flags.to(device)
            input_pert_flags = pert_flags[:, input_gene_ids]
        elif args.model_type =="drug":
            input_pert_flags = batch_data.drug_pert
        with torch.cuda.amp.autocast(enabled=args.amp):
            if args.model_type =="gene":
                output_dict = model(
                    mapped_input_gene_ids,
                    input_values,
                    input_pert_flags,
                    batch_data.pert_idx,
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=args.CLS,
                    CCE=args.CCE,
                    MVC=args.MVC,
                    ECS=args.ECS,
                )
                #差值
                output_values = output_dict["mlm_output"]
            elif args.model_type =="drug":
                _, _, _, _, _, output_dict = model(
                input_gene_ids,
                input_values,
                input_pert_flags,
                src_key_padding_mask=src_key_padding_mask,
                MVC=args.MVC,
                ECS=args.ECS,
                )
                output_values = output_dict["mlm_output"]

            masked_positions = torch.ones_like(
                input_values, dtype=torch.bool
            )  # Use all

            if args.model_type == "gene":
                loss = loss_mse = criterion(output_values, target_values, masked_positions)
            elif args.model_type == "drug":
                loss = loss_mse = drug_criterion(ctrl_expression[input_gene_ids],output_values, target_values)

            if args.explicit_zero_prob:
                loss_zero_log_prob = criterion_neg_log_bernoulli(
                    output_dict["mlm_zero_probs"], target_values, masked_positions
                )

                loss = loss + loss_zero_log_prob
        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            if len(w) > 0:
                logger.warning(
                    f"Found infinite gradient. This may be caused by the gradient "
                    f"scaler. The current scale is {scaler.get_scale()}. This warning "
                    "can be ignored if no longer occurs after autoscaling of the scaler."
                )
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_mse += loss_mse.item()
        if batch % args.log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / args.log_interval
            cur_loss = total_loss / args.log_interval
            cur_mse = total_mse / args.log_interval
            # ppl = math.exp(cur_loss)
            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} |"
            )
            total_loss = 0
            total_mse = 0
            start_time = time.time()
    return model


def evaluate(model: nn.Module, val_loader: torch.utils.data.DataLoader) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_loss = 0.0
    total_error = 0.0

    with torch.no_grad():
        for batch, batch_data in enumerate(val_loader):
            batch_size = len(batch_data.pert)
            batch_data.to(device)
            x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 2)
            x= x.view(-1)
            ori_gene_values =  x.view(batch_size, -1)
            # ori_gene_values = ctrl_expression.repeat(batch_size, 1).to(dtype=torch.float32).to(device)

            target_gene_values = batch_data.y.view(batch_size, n_genes)  # (batch_size, n_genes)

            if args.include_zero_gene in ["all", "batch-wise","highly_variable"]:
                if args.include_zero_gene == "all":
                    input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
                    if len(input_gene_ids) > args.max_seq_len:
                        input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[:args.max_seq_len]
     
                if args.include_zero_gene == "batch-wise":
                    input_gene_ids = (
                        ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                    )
                    if len(input_gene_ids) > args.max_seq_len:
                        input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[
                        :args.max_seq_len
                    ]
                
                if args.include_zero_gene == "highly_variable":
                    input_gene_ids = pert_data.highly_variable_gene_indices
                    input_gene_ids = torch.tensor(input_gene_ids, device=device)
                
                input_values = ori_gene_values[:, input_gene_ids]
                
                target_values = target_gene_values[:, input_gene_ids]

                mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
                mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

                src_key_padding_mask = torch.zeros_like(
                    input_values, dtype=torch.bool, device=input_values.device
                )
            
            if args.model_type =="gene":
                pert_flags = batch_data.pert_flags
                pert_flags = torch.tensor(pert_flags).long().view(batch_size, n_genes)
                pert_flags = pert_flags.to(device)
                input_pert_flags = pert_flags[:, input_gene_ids]
            elif args.model_type =="drug":
                input_pert_flags = batch_data.drug_pert
            with torch.cuda.amp.autocast(enabled=args.amp):
                if args.model_type =="gene":
                    output_dict = model(
                        mapped_input_gene_ids,
                        input_values,
                        input_pert_flags,
                        batch_data.pert_idx,
                        src_key_padding_mask=src_key_padding_mask,
                        CLS=args.CLS,
                        CCE=args.CCE,
                        MVC=args.MVC,
                        ECS=args.ECS,
                        do_sample=True,
                    )
                    #差值
                    output_values = output_dict["mlm_output"]
                elif args.model_type =="drug":
                    _, _, _, _, _, output_dict = model(
                    input_gene_ids,
                    input_values,
                    input_pert_flags,
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=args.CLS,
                    CCE=args.CCE,
                    MVC=args.MVC,
                    ECS=args.ECS,
                    do_sample=True,
                    )
                    output_values = output_dict["mlm_output"]
           
            
            masked_positions = torch.ones_like(
                input_values, dtype=torch.bool, device=input_values.device
            )

            if args.model_type == "gene":
                loss = loss_mse = criterion(output_values, target_values, masked_positions)
            elif args.model_type == "drug":
                loss = loss_mse = drug_criterion(ctrl_expression[input_gene_ids],output_values, target_values)
            if args.explicit_zero_prob:
                loss_zero_log_prob = criterion_neg_log_bernoulli(
                    output_dict["mlm_zero_probs"], target_values, masked_positions
                )
            
            
                loss = loss + loss_zero_log_prob

            total_loss += loss.item()
            total_error += masked_relative_error(
                output_values, target_values, masked_positions
            ).item()
    return total_loss / len(val_loader), total_error / len(val_loader)



def pred_perturb(
        model,
        batch_data,
        include_zero_gene="batch-wise",
        gene_ids=None,
        amp=True,
    ):
    """
    Args:
        batch_data: a dictionary of input data with keys.

    Returns:
        output Tensor of shape [N, seq_len]
    """
    model.eval()
   
    batch_size = len(batch_data.pert)
    x: torch.Tensor = batch_data.x
    x= x.view(-1)
    ori_gene_values = x.view(batch_size, -1)
    
    if include_zero_gene in ["all", "batch-wise","highly_variable"]:
        if include_zero_gene == "all":
            input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
            
        if include_zero_gene == "batch-wise":
            input_gene_ids = (
                ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
            )
        
        if include_zero_gene == "highly_variable":

            input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
        input_values = ori_gene_values[:, input_gene_ids]
        

        mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
        mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

        src_key_padding_mask = torch.zeros_like(
            input_values, dtype=torch.bool, device=device
        )
        if args.model_type =="gene":
            pert_flags = batch_data.pert_flags
            pert_flags = torch.tensor(pert_flags).long().view(batch_size, -1)
            pert_flags = pert_flags.to(device)
            input_pert_flags = pert_flags[:, input_gene_ids]

            
        elif args.model_type =="drug":
            input_pert_flags = batch_data.drug_pert
        with torch.cuda.amp.autocast(enabled=args.amp):
            if args.model_type =="gene":
                output_dict = model(
                    mapped_input_gene_ids,
                    input_values,
                    input_pert_flags,
                    batch_data.pert_idx,
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=False,
                    CCE=False,
                    MVC=False,
                    ECS=False,
                    do_sample=True,
                )
                #差值
                output_values = output_dict["mlm_output"]
            elif args.model_type =="drug":
                _, _, _, _, _, output_dict = model(
                input_gene_ids,
                input_values,
                input_pert_flags,
                src_key_padding_mask=src_key_padding_mask,
                CLS=False,
                CCE=False,
                MVC=False,
                ECS=False,
                do_sample=True,
                )
                output_values = output_dict["mlm_output"]
   
        output_values = output_dict["mlm_output"].float()
        pred_gene_values = torch.zeros_like(ori_gene_values)
        #差值
        pred_gene_values[:, input_gene_ids] = output_values
    return pred_gene_values,input_values

def predict(
    model: TransformerGenerator, pert_list: List[str],adata=None, pool_size: Optional[int] = None
) -> Dict:
    """
    Predict the gene expression values for the given perturbations.

    Args:
        model (:class:`torch.nn.Module`): The model to use for prediction.
        pert_list (:obj:`List[str]`): The list of perturbations to predict.
        pool_size (:obj:`int`, optional): For each perturbation, use this number
            of cells in the control and predict their perturbation results. Report
            the stats of these predictions. If `None`, use all control cells.
    """
    # adata = pert_data.adata
    ctrl_adata = adata[adata.obs["condition"] == "ctrl"]
    if pool_size is None:
        pool_size = len(ctrl_adata.obs)
    #gene_list = pert_data.gene_names.values.tolist()
    if args.model_type == "gene":
        gene_list = pert_data.pert_names
        for pert in pert_list:
            for i in pert:
                if i not in gene_list:
                    raise ValueError(
                        "The gene is not in the perturbation graph. Please select from GEARS.gene_list!"
                    )
    elif args.model_type == "drug":
        print("will create drug perturbation list dataset_for_prediction",pert_list)
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        results_pred = {}
        results_real = {}
        for pert in pert_list:
            if args.model_type == "gene":
                cell_graphs = create_cell_graph_dataset_for_prediction(
                pert_data.gene_pert_idx, pert, ctrl_adata, gene_list, device, num_samples=pool_size
                )
            elif args.model_type == "drug":
                cell_graphs = create_drug_cell_graph_dataset_for_prediction(
                 pert, ctrl_adata, device, num_samples=pool_size
                )
            loader = DataLoader(cell_graphs, batch_size=args.eval_batch_size, shuffle=False)
            preds = []
            reals = []
            for batch_data in loader:
                pred_gene_values ,input_values= pred_perturb(
                    model,batch_data , args.include_zero_gene, gene_ids=gene_ids, amp=args.amp, 
                )
                preds.append(pred_gene_values)
                reals.append(input_values)
            preds = torch.cat(preds, dim=0)
            reals = torch.cat(reals, dim=0)
                #preds.append(pred_gene_values.detach().cpu().numpy())
            
            results_pred["_".join(pert)] = preds.detach().cpu().numpy()
            results_real["_".join(pert)] = reals.detach().cpu().numpy()
    return results_pred,results_real

def plot_perturbation(
    model: nn.Module, query: str, save_file: str = None, pool_size: int = None,topnumber: str = None,figsize=[46.5, 4.5],cell_type=None
):
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt

    sns.set_theme(style="ticks", rc={"axes.facecolor": (0, 0, 0, 0)}, font_scale=1.5)
    
    adata = pert_data.adata
    if cell_type is not None:
        adata = adata[adata.obs["cell_type"]==cell_type]
    
    
    gene2idx = pert_data.node_map
    cond2name = dict(adata.obs[["condition", "condition_name"]].values)
    gene_raw2id = dict(zip(adata.var.index.values, adata.var.gene_name.values))
    
    if query not in cond2name.keys():
        return None
    de_idx = [
        gene2idx[gene_raw2id[i]]
        for i in adata.uns[topnumber][cond2name[query]]
    ]
    genes = [
        gene_raw2id[i] for i in adata.uns[topnumber][cond2name[query]]
    ]
    #.X.toarray
    try:
        truth = adata[adata.obs.condition == query].X.toarray()[:, de_idx]
    except:
        truth = adata[adata.obs.condition == query].X[:, de_idx]
    
    
    if  query.rpartition('+')[0]=="ctrl":
        pred,results_reals = predict(model, [[query.rpartition('+')[2]]], pool_size=None,adata=adata)
        reals_dict = {key: value.tolist() for key, value in results_reals.items()}
        serializable_dict = {key: value.tolist() for key, value in pred.items()}
        safe_query = query.replace("\\", "_").replace("/", "_").replace("*", "_")
        with open(f"{save_dir}/{cell_type}_{safe_query}.json", "w") as f:
            json.dump(serializable_dict, f)
        with open(f"{save_dir}/{cell_type}_{safe_query}real.json", "w") as f:
            json.dump(reals_dict , f)
        
        pred = pred[query.rpartition('+')[2]][:, de_idx]
    elif  query.rpartition('+')[2] == "ctrl":
        pred ,results_reals= predict(model, [[query.rpartition('+')[0]]], pool_size=None,adata=adata)
        reals_dict = {key: value.tolist() for key, value in results_reals.items()}
        serializable_dict = {key: value.tolist() for key, value in pred.items()}
        safe_query = query.replace("\\", "_").replace("/", "_").replace("*", "_")
        with open(f"{save_dir}/{cell_type}_{safe_query}.json", "w") as f:
            json.dump(serializable_dict, f)
        with open(f"{save_dir}/{cell_type}_{safe_query}real.json", "w") as f:
            json.dump(reals_dict , f)
        
        pred = pred[query.rpartition('+')[0]][:, de_idx]

    else:
        pred ,results_reals= predict(model, [query.split("+")], pool_size=None,adata=adata)
        reals_dict = {key: value.tolist() for key, value in results_reals.items()}
        serializable_dict = {key: value.tolist() for key, value in pred.items()}
        safe_query = query.replace("\\", "_").replace("/", "_").replace("*", "_")
        with open(f"{save_dir}/{cell_type}_{safe_query}.json", "w") as f:
            json.dump(serializable_dict, f)
        with open(f"{save_dir}/{cell_type}_{safe_query}real.json", "w") as f:
            json.dump(reals_dict , f)
        pred = pred["_".join(query.split("+"))][:, de_idx]
    ctrl_means = adata[adata.obs["condition"] == "ctrl"].to_df().mean()[de_idx].values

    pred = pred - ctrl_means
    truth = truth - ctrl_means

    
    plt.figure(figsize=figsize)
    plt.title(query)
    
    data = []
    for i in range(truth.shape[1]):
        truth_col = truth[:, i] # 获取truth的第i列
        pred_col = pred[:, i] # 获取pred的第i列
        data.append(truth_col)
        data.append(pred_col)
    
    data = np.array(data).T # 将data转换为numpy数组
    plt.boxplot(data, showfliers=True, medianprops=dict(linewidth=0))
    
    for i in range(pred.shape[1]):
        _ = plt.scatter((i+1)*2-1, np.mean(pred[:, i]), color="red")
    

    plt.axhline(0, linestyle="dashed", color="green")

    ax = plt.gca()
    newgenelabel=[]
    for i in genes:
       
        newgenelabel.append(str(i)+"real")
        newgenelabel.append(str(i)+"perd")

    ax.xaxis.set_ticklabels(np.array(newgenelabel), rotation=90)

    plt.ylabel("Change in Gene Expression over Control", labelpad=10)
    plt.tick_params(axis="x", which="major", pad=5)
    plt.tick_params(axis="y", which="major", pad=5)
    sns.despine() 
    if save_file:
        plt.savefig(save_file, bbox_inches="tight", transparent=False)


def perd_newperturbation(
    model: nn.Module, query: str,
):
   

    pred ,results_reals= predict(model, [query.split("+")], pool_size=1)
    reals_dict = {key: value.tolist() for key, value in results_reals.items()}
    serializable_dict = {key: value.tolist() for key, value in pred.items()}
    with open(f"{save_dir}/{query}.json", "w") as f:
        json.dump(serializable_dict, f)
    with open(f"{save_dir}/{query}real.json", "w") as f:
        json.dump(reals_dict , f)


def eval_perturb(
    loader: DataLoader, model: TransformerGenerator, device: torch.device
) -> Dict:
    """
    Run model in inference mode using a given data loader
    """
    model.eval()
    model.to(device)
    pert_cat = []
    pred = []
    truth = []
    pred_de = []
    ctrl_de = []
    ctrl_de20 = []
    ctrl_de50 = []
    ctrl_de100 = []
    ctrl_de200 = []
    ctrl_deall = []
    truth_de = []
    pred_de20 = []
    truth_de20 = []
    pred_de50 = []
    truth_de50 = []
    pred_de100 = []
    truth_de100 = []
    pred_de200 = []
    truth_de200 = []
    pred_deall = []
    truth_deall = []
    results = {}
    logvar = []

    for itr, batch in enumerate(loader):
        batch.to(device)
        pert_cat.extend(batch.pert)
        batch_size = len(batch.pert)
        with torch.no_grad():
            p,r = pred_perturb(model,batch, args.include_zero_gene, gene_ids=gene_ids)
            t = batch.y.view(batch_size, -1)
            #t = binning(t,device=device).to(device)
         

            pred.extend(p.cpu())
            truth.extend(t.cpu())

            # Differentially expressed genes
            for itr, de_idx in enumerate(batch.de_idx):
                pred_de.append(p[itr, de_idx])
                truth_de.append(t[itr, de_idx])
                ctrl_de.append(torch.tensor(ctrl_mean[de_idx]))
            
            for itr, de_idx in enumerate(batch.de_idx20):
                pred_de20.append(p[itr, de_idx])
                truth_de20.append(t[itr, de_idx])
                ctrl_de20.append(torch.tensor(ctrl_mean[de_idx]))
            
            for itr, de_idx in enumerate(batch.de_idx50):
                pred_de50.append(p[itr, de_idx])
                truth_de50.append(t[itr, de_idx])
                ctrl_de50.append(torch.tensor(ctrl_mean[de_idx]))

            for itr, de_idx in enumerate(batch.de_idx100):
                pred_de100.append(p[itr, de_idx])
                truth_de100.append(t[itr, de_idx])
                ctrl_de100.append(torch.tensor(ctrl_mean[de_idx]))
            
            for itr, de_idx in enumerate(batch.de_idx200):
                pred_de200.append(p[itr, de_idx])
                truth_de200.append(t[itr, de_idx])
                ctrl_de200.append(torch.tensor(ctrl_mean[de_idx]))

            for itr, de_idx in enumerate(batch.de_idxall):
                pred_deall.append(p[itr, de_idx])
                truth_deall.append(t[itr, de_idx])
                ctrl_deall.append(torch.tensor(ctrl_mean[de_idx]))

    # all genes
    results["pert_cat"] = np.array(pert_cat)
    pred = torch.stack(pred)
    truth = torch.stack(truth)
    results["pred"] = pred.detach().cpu().numpy().astype(np.float)
    results["truth"] = truth.detach().cpu().numpy().astype(np.float)

    pred_de = torch.stack(pred_de)
    truth_de = torch.stack(truth_de)
    ctrl_de = torch.stack(ctrl_de)

    pred_de20 = torch.stack(pred_de20)
    truth_de20 = torch.stack(truth_de20)
    ctrl_de20 = torch.stack(ctrl_de20)

    pred_de50 = torch.stack(pred_de50)
    truth_de50 = torch.stack(truth_de50)
    ctrl_de50 = torch.stack(ctrl_de50)

    pred_de100 = torch.stack(pred_de100)
    truth_de100 = torch.stack(truth_de100)
    ctrl_de100 = torch.stack(ctrl_de100)

    pred_de200 = torch.stack(pred_de200)
    truth_de200 = torch.stack(truth_de200)
    ctrl_de200 = torch.stack(ctrl_de200)


    results["pred_de"] = pred_de.detach().cpu().numpy().astype(np.float)
    results["truth_de"] = truth_de.detach().cpu().numpy().astype(np.float)
    results["ctrl_de"] = ctrl_de.detach().cpu().numpy().astype(np.float)

    results["pred_de20"] = pred_de20.detach().cpu().numpy().astype(np.float)
    results["truth_de20"] = truth_de20.detach().cpu().numpy().astype(np.float)
    results["ctrl_de20"] = ctrl_de20.detach().cpu().numpy().astype(np.float)

    results["pred_de50"] = pred_de50.detach().cpu().numpy().astype(np.float)
    results["truth_de50"] = truth_de50.detach().cpu().numpy().astype(np.float)
    results["ctrl_de50"] = ctrl_de50.detach().cpu().numpy().astype(np.float)

    results["pred_de100"] = pred_de100.detach().cpu().numpy().astype(np.float)
    results["truth_de100"] = truth_de100.detach().cpu().numpy().astype(np.float)
    results["ctrl_de100"] = ctrl_de100.detach().cpu().numpy().astype(np.float)

    results["pred_de200"] = pred_de200.detach().cpu().numpy().astype(np.float)
    results["truth_de200"] = truth_de200.detach().cpu().numpy().astype(np.float)
    results["ctrl_de200"] = ctrl_de200.detach().cpu().numpy().astype(np.float)

    return results



if __name__ == "__main__":
    set_seed(42)
    # settings for data prcocessing
    parser = argparse.ArgumentParser(description='Train and evaluate the model.')
    parser.add_argument('--pad_token', default='<pad>', type=str, help='Pad token')
    parser.add_argument('--pad_value', default=0, type=int, help='for padding values')
    parser.add_argument('--pert_pad_id', default=2, type=int, help='for perturbation padding values')
    parser.add_argument('--n_hvg', default=0, type=int, help='number of highly variable genes')
    parser.add_argument('--include_zero_gene', default="all", type=str, help='include zero expr genes in training input, "all", "batch-wise", "highly_variable", or False')
    parser.add_argument('--max_seq_len', default=1536, type=int, help='train max_seq_len')
    parser.add_argument('--MLM', default=True, type=lambda x: (str(x).lower() == 'true'), help='Whether to use masked language modeling, currently it is always on')
    parser.add_argument('--CLS', default=False, type=lambda x: (str(x).lower() == 'true'), help='Celltype classification objective')
    parser.add_argument('--CCE', default=False, type=lambda x: (str(x).lower() == 'true'), help='Contrastive cell embedding objective')
    parser.add_argument('--MVC', default=False, type=lambda x: (str(x).lower() == 'true'), help='Masked value prediction for cell embedding')
    parser.add_argument('--ECS', default=False, type=lambda x: (str(x).lower() == 'true'), help='Elastic cell similarity objective')
    parser.add_argument('--ecs_thres', default=0.0, type=float, help=' Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable')
    parser.add_argument('--pre_norm', default=False, type=lambda x: (str(x).lower() == 'true'), help='pre_norm')
    parser.add_argument('--cell_emb_style', default='cls', type=str, help='Cell embedding style')
    parser.add_argument('--mvc_decoder_style', default='inner product, detach', type=str, help='MVC decoder style')
    parser.add_argument('--amp', default=True, type=lambda x: (str(x).lower() == 'true'), help='Automatic mixed precision')
    parser.add_argument('--load_model', default="save/scGPT_human", type=str, help='Model path to load')# "save/scGPT_human"
    parser.add_argument('--load_param_prefixs', nargs='+', default=["encoder", "value_encoder", "transformer_encoder"], help='Prefixes of parameters to load')
    # ["encoder", "value_encoder", "transformer_encoder"]
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size for training')
    parser.add_argument('--eval_batch_size', default=64, type=int, help='Batch size for evaluation')
    parser.add_argument('--epochs', default=15, type=int, help='Number of epochs')
    parser.add_argument('--schedule_interval', default=1, type=int, help='Scheduler interval')
    parser.add_argument('--early_stop', default=5, type=int, help='Early stopping criteria')
    parser.add_argument('--n_GNN', default=1, type=int, help='Number of GNN layers')
    parser.add_argument('--embsize', default=512, type=int, help='Embedding dimension')
    parser.add_argument('--d_hid', default=512, type=int, help='Dimension of the feedforward network model in nn.TransformerEncoder')
    parser.add_argument('--nlayers', default=12, type=int, help='Number of nn.TransformerEncoderLayer in nn.TransformerEncoder')
    parser.add_argument('--nhead', default=8, type=int, help='Number of heads in nn.MultiheadAttention')
    parser.add_argument('--n_layers_cls', default=3, type=int, help='Number of layers for cell type classification')
    parser.add_argument('--dropout', default=0.2, type=float, help='Dropout probability')
    parser.add_argument('--use_fast_transformer', default=True, type=lambda x: (str(x).lower() == 'true'), help='Whether to use fast transformer')
    parser.add_argument('--explicit_zero_prob', default=False, type=lambda x: (str(x).lower() == 'true'), help='Explicit zero probability')
    parser.add_argument('--direction_lambda', default=0.1, type=float, help='Direction lambda')
    parser.add_argument('--log_interval', default=100, type=int, help='Logging interval')
    parser.add_argument('--rotary_emb_fraction', default=0.5, type=float, help='Rotary embedding fraction')
    parser.add_argument('--model_type', default="drug", type=str, help='gene or drug')
    parser.add_argument('--adata_file', default='/home/zqliu02/code/gene-drug-perturbation-model/data/kaggle_data_4celltype/perturb_processed.h5ad', type=str, help='Path to adata file')
    parser.add_argument('--screen_list', nargs='+', default= None, help='List of screens to plot')
    # ["ARHGAP22+ctrl","ATF4+ctrl","ATF6+ctrl","C7orf26+ctrl", "CAD+ctrl","CCND3+ctrl", "COPB1+ctrl", "EIF2AK3+ctrl", "ERN1+ctrl","GBF1+ctrl", "HSPA5+ctrl",  "IDH3A+ctrl", "IER3IP1+ctrl", "PPWD1+ctrl", "PSMD12+ctrl", "SOCS1+ctrl", "TIMM23+ctrl", "XBP1+ctrl","FECH+ctrl"]
    parser.add_argument('--perts_to_plot', nargs='+', default=None, help='Perturbations to plot')
    parser.add_argument('--data_name', default="kaggle_data_4celltype", type=str, help='Name of the dataset')
    parser.add_argument('--split', default="simulation", type=str, help='Split type for the dataset')
    parser.add_argument('--eval_perturb', default=True, type=lambda x: (str(x).lower() == 'true'), help='eval_perturb')
    args = parser.parse_args()

    special_tokens = [args.pad_token, "<cls>", "<eoc>"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(f"./save/dev_perturb_{args.data_name}-{time.strftime('%b%d-%H-%M')}/")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"saving to {save_dir}")
    logger = scg.logger
    scg.utils.add_file_handler(logger, save_dir / "run.log")
    # log running date and current git commit
    logger.info(f"Running on {time.strftime('%Y-%m-%d %H:%M:%S')}")
    pert_data = PertData("./data","delete",default_pert_graph=False)
    adata = sc.read_h5ad(args.adata_file)
   
    if args.screen_list is not None:
        adata = adata[~adata.obs["condition"].isin(args.screen_list)]

    pert_data.new_data_process(dataset_name = args.data_name, adata = adata,model_type = args.model_type)
    
    pert_data.prepare_split(split=args.split, seed=1)
    if args.perts_to_plot == None:
        args.perts_to_plot = pert_data.set2conditions['test']
    if args.model_type =='drug':
        pert_data.set2conditions['train'].remove("ctrl")


    print("perts_to_plot: ",args.perts_to_plot)
    print("loadend")

    pert_data.get_dataloader(batch_size=args.batch_size, test_batch_size=args.eval_batch_size)
    adata = pert_data.adata

    ctrl_expression = torch.tensor(
                np.mean(adata.X[adata.obs.condition == 'ctrl'],
                        axis=0)).reshape(-1, ).to(device)
    ctrl_mean = np.array(adata[adata.obs["condition"] == "ctrl"].to_df().mean().values)
    

    embsize = args.embsize
    nhead = args.nhead
    d_hid = args.d_hid
    nlayers = args.nlayers
    n_layers_cls = args.n_layers_cls
    dropout = args.dropout
    if args.load_model is not None:
        model_dir = Path(args.load_model)
        model_config_file = model_dir / "args.json"
        model_file = model_dir / "best_model.pt"
        vocab_file = model_dir / "vocab.json"
        vocab = GeneVocab.from_file(vocab_file)
        for s in special_tokens:
            if s not in vocab:
                vocab.append_token(s)
        pert_data.adata.var["id_in_vocab"] = [
            1 if gene in vocab else -1 for gene in pert_data.adata.var["gene_name"]
        ]
        gene_ids_in_vocab = np.array(pert_data.adata.var["id_in_vocab"])
        logger.info(
            f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
            f"in vocabulary of size {len(vocab)}."
        )
        genes = pert_data.adata.var["gene_name"].tolist()
        # model
        with open(model_config_file, "r") as f:
            model_configs = json.load(f)
        logger.info(
            f"Resume model from {model_file}, the model args will override the "
            f"config {model_config_file}."
        )
        embsize = model_configs["embsize"]
        nhead = model_configs["nhead"]
        d_hid = model_configs["d_hid"]
        nlayers = model_configs["nlayers"]
        n_layers_cls = model_configs["n_layers_cls"]
        dropout = model_configs["dropout"]

    else:
        genes = pert_data.adata.var["gene_name"].tolist()
        vocab = Vocab(
            VocabPybind(genes + special_tokens, None)
        )  # bidirectional lookup [gene <-> int]
    
    vocab.set_default_index(vocab["<pad>"])
    with open(save_dir / "vocab.json", "w") as f:
        json.dump(vocab.get_stoi(), f, indent=2)
    configs_args = vars(args) 
    with open(save_dir / "args.json", "w") as f:
        json.dump(configs_args, f, indent=4)
    gene_ids = np.array(
        [vocab[gene] if gene in vocab else vocab["<pad>"] for gene in genes], dtype=int
    )
    n_genes = len(genes)

    ntokens = len(vocab)  # size of vocabulary

   

    if args.model_type=="gene":
        num_perts = len(pert_data.pert_names)

        edge_list = get_similarity_network(network_type='go',
                                                    adata=pert_data.adata,
                                                    threshold=0.4,
                                                    k=20,
                                                    pert_list=pert_data.pert_names.tolist(),
                                                    data_path=pert_data.data_path,
                                                    data_name=pert_data.dataset_name,
                                                    split=pert_data.split, seed=pert_data.seed,
                                                    train_gene_set_size=pert_data.train_gene_set_size,
                                                    set2conditions= pert_data.set2conditions,
                                                    default_pert_graph=pert_data.default_pert_graph)
        edge_list = edge_list.dropna(subset=['source'])
        edge_list.reset_index(drop=True, inplace=True)
        edge_list = edge_list.dropna(subset=['target'])
        edge_list.reset_index(drop=True, inplace=True)

        sim_network = GeneSimNetwork(edge_list, pert_data.pert_names.tolist(), node_map = pert_data.node_map_pert)
        G_sim= sim_network.edge_index
        G_sim_weight = sim_network.edge_weight
        model = TransformerGenerator(
            ntokens,
            len(genes),
            embsize,
            nhead,
            d_hid,
            nlayers,
            nlayers_cls=n_layers_cls,
            n_cls=1,
            num_perts = num_perts,
            G_sim =G_sim,
            G_sim_weight = G_sim_weight,
            n_GNN=args.n_GNN,
            vocab=vocab,
            dropout=dropout,
            pad_token=args.pad_token,
            pad_value=args.pad_value,
            pert_pad_id=args.pert_pad_id,
            do_mvc=args.MVC,
            cell_emb_style=args.cell_emb_style,
            mvc_decoder_style=args.mvc_decoder_style,
            explicit_zero_prob = args.explicit_zero_prob,
            use_fast_transformer=args.use_fast_transformer,
        )
    elif args.model_type=="drug":
        model = TransformerModel(
            ntokens,
            embsize,
            nhead,
            d_hid,
            nlayers,
            vocab=vocab,
            dropout=dropout,
            pad_token=args.pad_token,
            pad_value=args.pad_value,
            do_mvc=args.MVC,
            do_dab=False,
            use_batch_labels=False,
            domain_spec_batchnorm=None,
            ecs_threshold=args.ecs_thres,
            explicit_zero_prob=args.explicit_zero_prob,
            use_fast_transformer=args.use_fast_transformer,
            pre_norm=args.pre_norm,
                )
    
    if args.load_param_prefixs is not None and args.load_model is not None:
        # only load params that start with the prefix
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file)
        pretrained_dict = {k : v for k, v in pretrained_dict.items() if any([k.startswith(prefix) for prefix in args.load_param_prefixs])}
        for k, v in pretrained_dict.items():
            logger.info(f"Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    elif args.load_model is not None:
        try:
            model.load_state_dict(torch.load(model_file))
            logger.info(f"Loading all model params from {model_file}")
        except:
            # only load params that are in the model and match the size
            model_dict = model.state_dict()
            pretrained_dict = torch.load(model_file)
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            for k, v in pretrained_dict.items():
                logger.info(f"Loading params {k} with shape {v.shape}")
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
    
    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=[0])
    criterion = masked_mse_loss
    drug_criterion = weighted_mse_loss
    criterion_cls = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.schedule_interval, gamma=0.9)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    best_val_loss = float("inf")
    best_model = None
    patience = 0
    # 查看显卡占用
    while True:
        memory_used = get_gpu_memory_usage()
        cuda_visible_devices = int(os.getenv('CUDA_VISIBLE_DEVICES', default=0))
        # 检查GPU的显存使用是否超过3GB
        if memory_used[cuda_visible_devices] > 3074:
            print(f"Video memory usage exceeds 3 GB, current usage: {memory_used[cuda_visible_devices]} MB, the program waits 10 seconds")
            time.sleep(10)
        else:
            print(f"Video memory usage is normal, current usage: {memory_used[cuda_visible_devices]} MB, continue executing the program")
            break
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_loader = pert_data.dataloader["train_loader"]
        valid_loader = pert_data.dataloader["val_loader"]

        model = train(
            model,
            train_loader,
        )
        val_loss, val_mre = evaluate(
            model,
            valid_loader,
        )
        elapsed = time.time() - epoch_start_time
        logger.info("-" * 89)
        logger.info(
            f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
            f"valid loss/mse {val_loss:5.4f} |"
        )
        logger.info("-" * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = None
            best_model = copy.deepcopy(model)
            logger.info(f"Best model with score {best_val_loss:5.4f}")
            patience = 0
        else:
            patience += 1
            if patience >= args.early_stop:
                logger.info(f"Early stop at epoch {epoch}")
                break
        torch.save(
            model.state_dict(),
            save_dir / f"model_{epoch}.pt",
        )

        scheduler.step()

    torch.save(best_model.state_dict(), save_dir / "best_model.pt")
    if args.eval_perturb:
        try:
            test_loader = pert_data.dataloader["test_loader"]
            torch.cuda.empty_cache()

            logger.info("start eval_perturb")
            test_res = eval_perturb(test_loader, best_model, device)
            logger.info("start compute_metrics")
            test_metrics, test_pert_res = compute_metrics(test_res)
            print(test_metrics)
            
            # save the dicts in json
            with open(f"{save_dir}/test_metrics.json", "w") as f:
                json.dump(test_metrics, f)
            with open(f"{save_dir}/test_pert_res.json", "w") as f:
                json.dump(test_pert_res, f)

            deeper_res = deeper_analysis(pert_data.adata, test_res)
            non_dropout_res = non_dropout_analysis(pert_data.adata, test_res)

            metrics = ["pearson_delta", "pearson_delta_de","pearson_delta_top20_de","pearson_top20_de","pearson_delta_top50_de","pearson_top50_de","pearson_delta_top100_de","pearson_top100_de","pearson_delta_top200_de","pearson_top200_de","mse_top20_de","mse_top50_de","mse_top100_de","mse_top200_de"]
            metrics_non_dropout = [
                "pearson_delta_top20_de_non_dropout",
                "pearson_top20_de_non_dropout",
            ]
            subgroup_analysis = {}
            for name in pert_data.subgroup["test_subgroup"].keys():
                subgroup_analysis[name] = {}
                for m in metrics:
                    subgroup_analysis[name][m] = []

                for m in metrics_non_dropout:
                    subgroup_analysis[name][m] = []

            for name, pert_list in pert_data.subgroup["test_subgroup"].items():
                for pert in pert_list:
                    for m in metrics:
                        subgroup_analysis[name][m].append(deeper_res[pert][m])

                    for m in metrics_non_dropout:
                        subgroup_analysis[name][m].append(non_dropout_res[pert][m])

            for name, result in subgroup_analysis.items():
                for m in result.keys():
                    subgroup_analysis[name][m] = np.mean(subgroup_analysis[name][m])
                    logger.info("test_" + name + "_" + m + ": " + str(subgroup_analysis[name][m]))
        except:
            print("eval_perturb ERR")
        torch.cuda.empty_cache()
        
    
    logger.info("start plot_perturbation")
    for cell_type in  pert_data.cell_type_dict.keys():
        for p in args.perts_to_plot:
            safe_query = p.replace("\\", "_").replace("/", "_").replace("*", "_")
            plot_perturbation(best_model, p, pool_size=100, save_file=f"{save_dir}/{cell_type}_{safe_query}_de20.png", topnumber = 'top_non_zero_de_20',figsize=[16.5, 4.5],cell_type=cell_type)
