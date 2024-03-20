# Function: evaluate the performance for perturbed response prediction.

import argparse
import pickle
import re
import os
import numpy as np
import pandas as pd
import scipy
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import anndata as ad
import scanpy as sc
import gseapy as gp
from sklearn.linear_model import LogisticRegression

# Define functions.
def retrieve_DE(controlCombined_dataset, dataset_name, condition_list_pred):
    '''Retrieve DE gene as selected gene for truth and prediction datasets with detail information.
      
    The modified version of sc.tl.rank_genes_groups is used here.
    In the modified version, pseudocount is changed from 1e-9 to 1 when calculate logfoldchanges.
    The path of modified version of sc.tl.rank_genes_groups is:
    /home/yzhao02/miniconda3/lib/python3.11/site-packages/scanpy/tools/_rank_genes_groups.py'''

    global dataset__DEinfo
    global infoName__infoValue

    sc.tl.rank_genes_groups(controlCombined_dataset, groupby = "condition", reference = "ctrl", rankby_abs = True, use_raw = False)
    DE_df = sc.get.rank_genes_groups_df(controlCombined_dataset, group = None)
    DE_df["logFC_abs"] = [abs(x) for i,x in enumerate(DE_df.logfoldchanges)]
    DE_df.sort_values("logFC_abs", ascending = False, ignore_index = True, inplace = True)

    dataset__DEinfo[dataset_name] = DE_df

    for each_topNum in (20, 50, 80, 100):
        for each_condition in condition_list_pred:
            DE_df_selected = DE_df.loc[DE_df.group == each_condition, :]
            DE_gene = list(DE_df_selected.names)[:each_topNum]
            DE_logFC = list(DE_df_selected.logfoldchanges)[:each_topNum]
            DE_logFC_abs = list(DE_df_selected.logFC_abs)[:each_topNum]
            DE_logFC_abs = [abs(each_DE_logFC) for each_DE_logFC in DE_logFC]
            DE_adjPval = list(DE_df_selected.pvals_adj)[:each_topNum]

            infoName__infoValue["dataset_name"] += [dataset_name] * each_topNum
            infoName__infoValue["topNum"] += [each_topNum] * each_topNum
            infoName__infoValue["condition"] += [each_condition] * each_topNum
            infoName__infoValue["DE_gene"] += DE_gene
            infoName__infoValue["DE_logFC"] += DE_logFC
            infoName__infoValue["DE_logFC_abs"] += DE_logFC_abs
            infoName__infoValue["DE_adjPval"] += DE_adjPval

def retrieve_DE_deltaExpr(info_df, control, truth, pred, condition_list_pred):
    '''Retrieve delta expression profile (truth/prediction - control) of top DE genes for datasets in this or other conditions'''

    global exprName__exprValue

    for each_topNum in (20, 50, 80, 100):
        DE_set = info_df.loc[(info_df.dataset_name == "truth") & (info_df.topNum == each_topNum), :]
        
        for each_condition in condition_list_pred:
            DE_gene = list(DE_set.loc[DE_set.condition == each_condition, "DE_gene"])
            condition_list_pred_copy = condition_list_pred.copy()

            for each_condition_copy in condition_list_pred_copy:
                truth_selected = np.mean(truth[truth.obs.condition == each_condition_copy, DE_gene].X.toarray(), axis = 0)
                pred_selected = np.mean(pred[pred.obs.condition == each_condition_copy, DE_gene].X.toarray(), axis = 0)
                control_selected = np.mean(control[control.obs.condition == "ctrl", DE_gene].X.toarray(), axis = 0)
                
                truth_delta = truth_selected - control_selected
                pred_delta = pred_selected - control_selected

                exprName__exprValue["dataset_name"].append("truth")
                exprName__exprValue["delta_expr"].append(truth_delta)
                exprName__exprValue["dataset_name"].append("pred")
                exprName__exprValue["delta_expr"].append(pred_delta)
                exprName__exprValue["topNum"] += [each_topNum] * 2
                exprName__exprValue["condition_DE"] += [each_condition] * 2
                exprName__exprValue["condition_expr"] += [each_condition_copy] * 2

def vis_between_PCC(deltaExpr_df, condition_list_pred):
    '''Visualize Pearson correlation coefficient of delta expression profile of top DE between datasets of truth and prediction.'''
    
    name__value = {}
    name__value["topNum"] = []
    name__value["class"] = []
    name__value["condition"] = []
    name__value["PCC"] = []

    for each_topNum in (20, 50, 80, 100):
        DE_set = deltaExpr_df.loc[deltaExpr_df.topNum == each_topNum, :]
        
        for each_condition in condition_list_pred:
            truth_delta = list(DE_set.loc[(DE_set.dataset_name == "truth") & 
                                          (DE_set.condition_DE == each_condition) & 
                                          (DE_set.condition_expr == each_condition), 
                                          "delta_expr"])[0]
            pred_delta = list(DE_set.loc[(DE_set.dataset_name == "pred") & 
                                         (DE_set.condition_DE == each_condition) & 
                                         (DE_set.condition_expr == each_condition), 
                                         "delta_expr"])[0]
            
            PCC = pearsonr(truth_delta, pred_delta)
            PCC = PCC.statistic

            name__value["topNum"].append(each_topNum)
            name__value["class"].append("target")
            name__value["condition"].append(each_condition)
            name__value["PCC"].append(PCC)

            other_list = condition_list_pred.copy()
            other_list.remove(each_condition)
            
            other_PCC = []

            for each_other in other_list:
                truth_delta = list(DE_set.loc[(DE_set.dataset_name == "truth") & 
                                              (DE_set.condition_DE == each_other) & 
                                              (DE_set.condition_expr == each_other), 
                                              "delta_expr"])[0]
                pred_delta = list(DE_set.loc[(DE_set.dataset_name == "pred") &
                                             (DE_set.condition_DE == each_other) &
                                             (DE_set.condition_expr == each_condition),
                                             "delta_expr"])[0]
                
                PCC = pearsonr(truth_delta, pred_delta)
                PCC = PCC.statistic
                other_PCC.append(PCC)

            other_PCC_mean = np.mean(other_PCC)

            name__value["topNum"].append(each_topNum)
            name__value["class"].append("control")
            name__value["condition"].append(each_condition)
            name__value["PCC"].append(other_PCC_mean)

    plot_df = pd.DataFrame(name__value)    
    plot_df.sort_values(by = ["topNum", "class", "PCC"], ignore_index = True, inplace = True)
    plot_df.to_csv(os.path.join(output_dir, "DE_gene_between_PCC.tsv"), header = True, index = False, sep = "\t")

    for each_topNum in (20, 50, 80, 100):
        plot_df_selected = plot_df.loc[plot_df.topNum == each_topNum, :]
        
        target_median = round(np.median(plot_df_selected.loc[plot_df_selected["class"] == "target", "PCC"]), 2)
        control_median = round(np.median(plot_df_selected.loc[plot_df_selected["class"] == "control", "PCC"]), 2)

        target_name = ": ".join(["target", str(target_median)])
        control_name = ": ".join(["control", str(control_median)])

        plot_df_selected["class"] = [re.sub("target", target_name, x) for x in plot_df_selected["class"]]
        plot_df_selected["class"] = [re.sub("control", control_name, x) for x in plot_df_selected["class"]]

        fig, ax = plt.subplots()
        sns_plot = sns.displot(plot_df_selected, x = "PCC", hue = "class", kind = "kde", common_norm = False, 
                               height = 6, aspect = 1, palette = {target_name:"#4c72b0", control_name:"#dd8452"})
        sns_plot.refline(x = target_median, color = "#4c72b0")
        sns_plot.refline(x = control_median, color = "#dd8452")
        plt.title("The PCC of delta expr of top %s DE \n between truth and prediction" % str(each_topNum), y = 0.9, fontsize = 10)
        plt.savefig(os.path.join(output_dir, "top_%s_DE_between_PCC.png" % str(each_topNum)))
        plt.clf()
        plt.close()

def vis_within_PCC(deltaExpr_df, condition_list_pred):
    '''Visualize Pearson correlation coefficient of delta expression profile of top DE within datasets of truth or prediction.'''
    
    name__value = {}
    name__value["topNum"] = []
    name__value["class"] = []
    name__value["condition"] = []
    name__value["PCC"] = []

    for each_topNum in (20, 50, 80, 100):
        DE_set = deltaExpr_df.loc[deltaExpr_df.topNum == each_topNum, :]

        for each_condition in condition_list_pred:
            other_list = condition_list_pred.copy()
            other_list.remove(each_condition)
            
            other_PCC_truth = []
            other_PCC_pred = []

            for each_other in other_list:
                truth_delta_this = list(DE_set.loc[(DE_set.dataset_name == "truth") & 
                                                   (DE_set.condition_DE == each_condition) & 
                                                   (DE_set.condition_expr == each_condition), 
                                                   "delta_expr"])[0]
                truth_delta_that = list(DE_set.loc[(DE_set.dataset_name == "truth") & 
                                                   (DE_set.condition_DE == each_condition) & 
                                                   (DE_set.condition_expr == each_other), 
                                                   "delta_expr"])[0]
                
                PCC_truth = pearsonr(truth_delta_this, truth_delta_that)
                PCC_truth = PCC_truth.statistic
                other_PCC_truth.append(PCC_truth)
                
                pred_delta_this = list(DE_set.loc[(DE_set.dataset_name == "pred") & 
                                                  (DE_set.condition_DE == each_condition) & 
                                                  (DE_set.condition_expr == each_condition), 
                                                  "delta_expr"])[0]
                pred_delta_that = list(DE_set.loc[(DE_set.dataset_name == "pred") & 
                                                  (DE_set.condition_DE == each_condition) & 
                                                  (DE_set.condition_expr == each_other), 
                                                  "delta_expr"])[0]

                PCC_pred = pearsonr(pred_delta_this, pred_delta_that)
                PCC_pred = PCC_pred.statistic
                other_PCC_pred.append(PCC_pred)

            other_PCC_truth_mean = np.mean(other_PCC_truth)
            other_PCC_pred_mean = np.mean(other_PCC_pred)

            name__value["topNum"] += [each_topNum] * 2
            name__value["class"] += ["truth", "pred"]
            name__value["condition"] += [each_condition] * 2
            name__value["PCC"] += [other_PCC_truth_mean, other_PCC_pred_mean]
               
    plot_df = pd.DataFrame(name__value)    
    plot_df.sort_values(by = ["topNum", "class", "PCC"], ignore_index = True, inplace = True)
    plot_df.to_csv(os.path.join(output_dir, "DE_gene_within_PCC.tsv"), header = True, index = False, sep = "\t")

    for each_topNum in (20, 50, 80, 100):
        plot_df_selected = plot_df.loc[plot_df.topNum == each_topNum, :]
        
        truth_median = round(np.median(plot_df_selected.loc[plot_df_selected["class"] == "truth", "PCC"]), 2)
        pred_median = round(np.median(plot_df_selected.loc[plot_df_selected["class"] == "pred", "PCC"]), 2)

        truth_name = ": ".join(["truth", str(truth_median)])
        pred_name = ": ".join(["pred", str(pred_median)])

        plot_df_selected["class"] = [re.sub("truth", truth_name, x) for x in plot_df_selected["class"]]
        plot_df_selected["class"] = [re.sub("pred", pred_name, x) for x in plot_df_selected["class"]]

        fig, ax = plt.subplots()
        sns_plot = sns.displot(plot_df_selected, x = "PCC", hue = "class", kind = "kde", common_norm = False, 
                               height = 6, aspect = 1, palette = {truth_name:"#4c72b0", pred_name:"#dd8452"})
        sns_plot.refline(x = truth_median, color = "#4c72b0")
        sns_plot.refline(x = pred_median, color = "#dd8452")
        plt.title("The PCC of delta expr of top %s DE \n within truth and prediction" % str(each_topNum), y = 0.9, fontsize = 10)
        plt.savefig(os.path.join(output_dir, "top_%s_DE_within_PCC.png" % str(each_topNum)))
        plt.clf()
        plt.close()

def vis_JaccardIndex(info_df, condition_list_pred):
    '''Visualize Jaccard index of top DE genes between datasets of truth and prediction.'''

    name__value = {}
    name__value["topNum"] = []
    name__value["condition"] = []
    name__value["JaccardIndex"] = []

    for each_topNum in (20, 50, 80, 100):
        DE_set = info_df.loc[(info_df.dataset_name.isin(["truth", "pred"])) & (info_df.topNum == each_topNum), :]
        
        for each_condition in condition_list_pred:
            truth_gene = list(DE_set.loc[(DE_set.dataset_name == "truth") & (DE_set.condition == each_condition), "DE_gene"])
            pred_gene = list(DE_set.loc[(DE_set.dataset_name == "pred") & (DE_set.condition == each_condition), "DE_gene"])

            overlapped_num = float(len(set(truth_gene) & set(pred_gene)))
            union_num = float(len(set(truth_gene + pred_gene)))
            JaccardIndex = round(overlapped_num / union_num, 2)
            
            name__value["topNum"].append(each_topNum)
            name__value["condition"].append(each_condition)
            name__value["JaccardIndex"].append(JaccardIndex)

        plot_df = pd.DataFrame(name__value)
        
        fig, ax = plt.subplots(figsize = (8,16))
        sns.barplot(data = plot_df, x = "JaccardIndex", y = "condition", hue = "topNum", orient = "h", ax = ax)
        
        for i in ax.containers:
            ax.bar_label(i,)
        
        plt.title("Jaccard index of DE genes between truth and prediction")
        plt.xlabel("")
        plt.ylabel("")
        plt.savefig(os.path.join(output_dir, "DE_gene_Jaccard_index.png"))
        plt.clf()
        plt.close()

def vis_prevalence(info_df):
    '''Visualize the prevalence of DE genes among all conditions in truth and prediction datasets.'''

    name__value = {}
    name__value["topNum"] = []
    name__value["dataset_name"] = []
    name__value["DE_gene"] = []
    name__value["prevalence"] = []
    name__value["DE_logFC_mean"] = []

    for each_topNum in (20, 50, 80, 100):
        for each_dataset in ("truth", "pred"):
            DE_set = info_df.loc[(info_df.dataset_name == each_dataset) & (info_df.topNum == each_topNum), 
                                 ["condition", "DE_gene", "DE_logFC"]]
            DE__count = {}
            DE__logFC = {}
          
            for each_DE in list(DE_set.DE_gene):
                if not each_DE in DE__count.keys():
                    DE__count[each_DE] = 0
                DE__count[each_DE] += 1

            for each_DE in DE__count.keys():
                DE__logFC[each_DE] = list(DE_set.loc[DE_set.DE_gene == each_DE, "DE_logFC"])

            for each_DE in DE__count.keys():
                name__value["topNum"].append(each_topNum)
                name__value["dataset_name"].append(each_dataset)
                name__value["DE_gene"].append(each_DE)
                each_prevalence = round(float(DE__count[each_DE]) / float(len(condition_list_pred)), 2)
                name__value["prevalence"].append(each_prevalence)
                name__value["DE_logFC_mean"].append(round(np.mean(DE__logFC[each_DE]), 2))

    plot_df = pd.DataFrame(name__value)
    plot_df.sort_values(by = ["topNum", "dataset_name", "prevalence"], ignore_index = True, inplace = True)
    plot_df.to_csv(os.path.join(output_dir, "DE_gene_prevalence_info.tsv"), header = True, index = False, sep = "\t")
    
    for each_topNum in (20, 50, 80, 100):
        plot_df_selected = plot_df.loc[plot_df.topNum == each_topNum, :]
        
        truth_median = round(np.median(plot_df_selected.loc[plot_df_selected.dataset_name == "truth", "prevalence"]), 2)
        pred_median = round(np.median(plot_df_selected.loc[plot_df_selected.dataset_name == "pred", "prevalence"]), 2)

        truth_name = ": ".join(["truth", str(truth_median)])
        pred_name = ": ".join(["pred", str(pred_median)])
        
        plot_df_selected["dataset_name"] = [re.sub("truth", truth_name, x) for x in plot_df_selected["dataset_name"]]
        plot_df_selected["dataset_name"] = [re.sub("pred", pred_name, x) for x in plot_df_selected["dataset_name"]]

        sns_plot = sns.displot(plot_df_selected, x = "prevalence", hue = "dataset_name", kind = "kde", common_norm = False, 
                               height = 6, aspect = 1, palette = {truth_name:"#4c72b0", pred_name:"#dd8452"})
        sns_plot.refline(x = truth_median, color = "#4c72b0")
        sns_plot.refline(x = pred_median, color = "#dd8452")
        plt.title("The prevalence of top %s DE genes" % str(each_topNum), y = 0.9, fontsize = 10)
        plt.savefig(os.path.join(output_dir, "top_%s_DE_gene_prevalence.png" % str(each_topNum)))
        plt.clf()
        plt.close()

def perform_funcEnrich(info_df, condition_list_pred):
    '''Perform functional enrichment analysis for top 20 DE genes of each condition in truth and prediction datasets.'''
    
    enrich_result_all = pd.DataFrame()

    for each_dataset in ("truth", "pred"):
        DE_set = info_df.loc[(info_df.dataset_name == each_dataset) & (info_df.topNum == 20), 
                             ["condition", "DE_gene"]]
        for each_condition in condition_list_pred:
            DE_gene = list(DE_set.loc[DE_set.condition == each_condition, "DE_gene"])
            
            enrich_result = gp.enrichr(gene_list = DE_gene, gene_sets = ["GO_Biological_Process_2023"], organism = "Human")
            enrich_result = enrich_result.results.iloc[:20, :]
            enrich_result["condition_dataset"] = "_".join([each_condition, each_dataset])

            enrich_result_all = pd.concat([enrich_result_all, enrich_result], ignore_index = True)

    first_column = enrich_result_all.pop("condition_dataset")
    enrich_result_all.insert(0, "condition_dataset", first_column)

    enrich_result_all.sort_values(by = ["condition_dataset"], ignore_index = True, inplace = True)
    enrich_result_all.to_csv(os.path.join(output_dir, "top_20_DE_gene_funcEnriched.tsv"), header = True, index = False, sep = "\t")

def vis_DE_rank(info_df, dataset__DEinfo, condition_list_pred):
    '''Visualize rank by absolute logFC of top DE genes identified from truth in prediction dataset and vice versa.'''
    
    name__value = {}
    name__value["topNum"] = []
    name__value["class"] = []
    name__value["condition"] = []
    name__value["rank"] = []
    
    truth_DEinfo = dataset__DEinfo["truth"]
    pred_DEinfo = dataset__DEinfo["pred"]

    for each_topNum in (20, 50, 80, 100):
        DE_set = info_df.loc[(info_df.dataset_name.isin(["truth", "pred"])) & (info_df.topNum == each_topNum), :]
        
        for each_condition in condition_list_pred:
            truth_DE = DE_set.loc[(DE_set.dataset_name == "truth") & (DE_set.condition == each_condition), "DE_gene"]
            pred_DE = DE_set.loc[(DE_set.dataset_name == "pred") & (DE_set.condition == each_condition), "DE_gene"]
            
            truth_rank = []
            pred_rank = []
            
            for each_truth_DE in truth_DE:
                pred_rank.append(list(pred_DEinfo.loc[pred_DEinfo.group == each_condition, "names"]).index(each_truth_DE) + 1)
           
            for each_pred_DE in pred_DE:
                truth_rank.append(list(truth_DEinfo.loc[truth_DEinfo.group == each_condition, "names"]).index(each_pred_DE) + 1)
    
            name__value["topNum"] += [each_topNum] * each_topNum * 2
            name__value["class"] += ["truth"] * each_topNum + ["pred"] * each_topNum
            name__value["condition"] += [each_condition] * each_topNum * 2
            name__value["rank"] += truth_rank + pred_rank
 
    plot_df = pd.DataFrame(name__value)
    plot_df.sort_values(by = ["topNum", "class"], ignore_index = True, inplace = True)
    plot_df.to_csv(os.path.join(output_dir, "DE_gene_rank_dist.tsv"), header = True, index = False, sep = "\t")

    for each_topNum in (20, 50, 80, 100):
        plot_df_selected = plot_df.loc[plot_df.topNum == each_topNum, :]

        fig, ax = plt.subplots(figsize = (12, 8))
        sns.boxplot(data = plot_df_selected, x = "condition", y = "rank", hue = "class", ax = ax)
        sns.move_legend(ax, "upper left", bbox_to_anchor = (1,1))
        plt.title("Rank distribution of top %s DE" % str(each_topNum))
        plt.xticks(rotation = 90)
        plt.xlabel("")
        plt.savefig(os.path.join(output_dir, "top_%s_DE_rank_dist.png" % str(each_topNum)), bbox_inches = "tight")
        plt.clf()
        plt.close()

def vis_perturbedGene_rank(dataset__DEinfo, condition_list_pred):
    '''Visualize rank by absolute logFC of perturbed genes.'''
    
    name__value = {}
    name__value["class"] = []
    name__value["condition"] = []
    name__value["rank"] = []

    truth_DEinfo = dataset__DEinfo["truth"]
    pred_DEinfo = dataset__DEinfo["pred"]

    for each_condition in condition_list_pred:
        truth_rank = list(truth_DEinfo.loc[truth_DEinfo.group == each_condition, "names"]).index(each_condition) + 1
        pred_rank = list(pred_DEinfo.loc[pred_DEinfo.group == each_condition, "names"]).index(each_condition) + 1

        name__value["condition"] += [each_condition] * 2
        name__value["class"] += ["truth", "pred"]
        name__value["rank"] += [truth_rank, pred_rank]

    plot_df = pd.DataFrame(name__value)
    plot_df.sort_values(by = ["condition", "class"], ignore_index = True, inplace = True)
    plot_df.to_csv(os.path.join(output_dir, "perturbed_gene_rank.tsv"), header = True, index = False, sep = "\t")

    fig, ax = plt.subplots(figsize = (12,8))
    sns.barplot(data = plot_df, x = "condition", y = "rank", hue = "class", ax = ax)
    for i in ax.containers:
        ax.bar_label(i,)
    plt.title("Rank of perturbed genes in truth and prediction dataset")
    plt.xticks(rotation = 90)
    plt.xlabel("")
    plt.ylabel("")
    plt.savefig(os.path.join(output_dir, "perturbed_gene_rank.png"))
    plt.clf()
    plt.close()

def vis_logistic_regression(truth, pred):
    '''Visualize the result of logistic regression analysis on prediction dataset.'''

    truth_expr = truth.X.toarray()
    truth_label = truth.obs.condition.tolist()
    
    pred_expr = pred.X.toarray()
    pred_label = pred.obs.condition.tolist()
    
    clf = LogisticRegression(penalty = "l2", multi_class = "ovr", random_state = 0).fit(truth_expr, truth_label)
    proba_result = clf.predict_proba(pred_expr)
    
    proba_result = pd.DataFrame(proba_result)
    proba_result.columns = clf.classes_
    proba_result["condition"] = pred_label
    
    mean_list = []
    for each_condition in proba_result.columns[:-1]:
        each_mean = np.mean(proba_result.loc[proba_result.condition == each_condition, :].iloc[:, :-1], axis = 0)
        mean_list.append(each_mean)
    
    mean_array = np.array(mean_list)
    mean_array_df = pd.DataFrame(mean_array)
    mean_array_df.index = proba_result.columns[:-1]
    mean_array_df.columns = proba_result.columns[:-1]
    
    fig, ax = plt.subplots(figsize = (25, 15))
    ax = sns.heatmap(mean_array_df, cmap = "PuBu", vmin = 0, annot = True, fmt = ".2g")
    column_max = mean_array_df.idxmax(axis = 0)
    for col_pos, col_name in enumerate(mean_array_df.columns):
        row_pos = mean_array_df.index.get_loc(column_max[col_name])
        ax.add_patch(Rectangle((col_pos, row_pos), 1, 1, fill = False, edgecolor = "red", lw = 3))
    plt.xticks(rotation = 90)
    plt.yticks(rotation = 90)
    plt.savefig(os.path.join(output_dir, "prediction_logistic_regression.png"))

# The __main__ function.
if __name__ == "__main__":
    
    # Parse arguments.
    parser = argparse.ArgumentParser(description = "Evaluate the performance for perturbed response prediction model.")
    help_desc = "The input h5ad file recording expression profile of control cells, required"
    parser.add_argument("--input_h5ad", default = "save/dev_perturb_k562_compound188_1200_bin-Jan12-11-25/h5ad", type = str, help = help_desc)
    help_desc = "The input h5ad file recording expression profile of perturbed cells in prediction, required"
    parser.add_argument("--dataset_name", default = "k562_compound188_1200_bin", type = str, help = help_desc)
    help_desc = "The output directory, required"
    parser.add_argument("--output_dir", default = "save/dev_perturb_k562_compound188_1200_bin-Jan12-11-25/result", type = str, help = help_desc)
    parser.add_argument("--model_type", default = "drug", type = str)
    args = parser.parse_args()
    input_h5ad_control = args.input_h5ad + "/ctrl.h5ad"
    input_h5ad_pred = args.input_h5ad + "/scGPTpred.h5ad"
    input_h5ad_truth = args.input_h5ad + "/testtruth.h5ad"
    dataset_name = args.dataset_name
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
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
    	
    truth = sc.read_h5ad(input_h5ad_truth)

    # truth.var.index = truth.var["gene_name"]
    
    truth.obs_names_make_unique()
    truth.obs.condition = [re.sub("\+ctrl", "", x) for i,x in enumerate(truth.obs.condition)]
    # Combine input h5ad files.
    control_truth = ad.concat([control, truth], index_unique = "-")
    control_pred = ad.concat([control, pred], index_unique = "-")

    # Retrieve perturbation condition list in prediction datasets.
    condition_list_pred = list(set(pred.obs.condition))

    # Initialize dictionaries used to store top DE genes and their information.
    infoName__infoValue = {}
    infoName__infoValue["dataset_name"] = []
    infoName__infoValue["topNum"] = []
    infoName__infoValue["condition"] = []
    infoName__infoValue["DE_gene"] = []
    infoName__infoValue["DE_logFC"] = []
    infoName__infoValue["DE_logFC_abs"] = []
    infoName__infoValue["DE_adjPval"] = []

    dataset__DEinfo = {}

    # Retrieve and store top DE genes and their information into initialized dictionaries.
    if os.path.exists(os.path.join(output_dir, "DE_gene_info.pickle")) and os.path.exists(os.path.join(output_dir, "DE_gene_info.raw.pickle")): 
        with open(os.path.join(output_dir, "DE_gene_info.pickle"), "rb") as input_file:
            info_df = pickle.load(input_file)
        
        with open(os.path.join(output_dir, "DE_gene_info.raw.pickle"), "rb") as input_file:
            dataset__DEinfo = pickle.load(input_file)
    else:
        print(retrieve_DE.__doc__) 
        retrieve_DE(control_truth, "truth", condition_list_pred)
        retrieve_DE(control_pred, "pred", condition_list_pred)
        
        info_df = pd.DataFrame(infoName__infoValue)
        
        info_df.to_csv(os.path.join(output_dir, "DE_gene_info.tsv"), header = True, index = False, sep = "\t")
        
        with open(os.path.join(output_dir, "DE_gene_info.pickle"), "wb") as output_file:
            pickle.dump(info_df, output_file)
    
        with open(os.path.join(output_dir, "DE_gene_info.raw.pickle"), "wb") as output_file:
            pickle.dump(dataset__DEinfo, output_file)

    # Initialize the dictionary used to store delta expression profile of top DE genes.
    exprName__exprValue = {}
    exprName__exprValue["dataset_name"] = []
    exprName__exprValue["topNum"] = []
    exprName__exprValue["condition_DE"] = []
    exprName__exprValue["condition_expr"] = []
    exprName__exprValue["delta_expr"] = []

    # Retrieve and store delta expression profile of top DE genes into the initialized dictionary.
    if os.path.exists(os.path.join(output_dir, "DE_gene_deltaExpr.pickle")):
        with open(os.path.join(output_dir, "DE_gene_deltaExpr.pickle"), "rb") as input_file:
            deltaExpr_df = pickle.load(input_file)
    else:
        print(retrieve_DE_deltaExpr.__doc__)
        retrieve_DE_deltaExpr(info_df, control, truth, pred, condition_list_pred)
        
        deltaExpr_df = pd.DataFrame(exprName__exprValue) 
        
        with open(os.path.join(output_dir, "DE_gene_deltaExpr.pickle"), "wb") as output_file:
            pickle.dump(deltaExpr_df, output_file)

    # Visualize PCC of delta expr of top DE genes between truth and prediction datasets.
    print(vis_between_PCC.__doc__) 
    vis_between_PCC(deltaExpr_df, condition_list_pred)

    # Visualize PCC of delta expr of top DE genes within truth or prediction dataset.
    print(vis_within_PCC.__doc__)
    vis_within_PCC(deltaExpr_df, condition_list_pred)

    # Visualize Jaccard index of top DE genes.
    print(vis_JaccardIndex.__doc__)
    vis_JaccardIndex(info_df, condition_list_pred)

    # Visualize the prevalence of top DE genes.
    print(vis_prevalence.__doc__)
    vis_prevalence(info_df)
    
    
    # Perform functional enrichment analysis for top 20 DE genes.
    print(perform_funcEnrich.__doc__)
    perform_funcEnrich(info_df, condition_list_pred)

    # Visualize the rank distribution of top DE genes.
    print(vis_DE_rank.__doc__)
    vis_DE_rank(info_df, dataset__DEinfo, condition_list_pred)


    if args.model_type == "gene":
    # Visualize the rank of perturbed genes.
        print(vis_perturbedGene_rank.__doc__)
        vis_perturbedGene_rank(dataset__DEinfo, condition_list_pred)

    # Visualize the result of logistic regression analysis on prediction dataset.
    print(vis_logistic_regression.__doc__)
    vis_logistic_regression(truth, pred)
