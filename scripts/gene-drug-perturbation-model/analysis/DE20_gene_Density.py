import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns
import argparse
import pickle

if __name__ == "__main__":

    
    parser = argparse.ArgumentParser(description='evaluate the model.')
    parser.add_argument('--savedir1', default='/home/zqliu02/code/gene-drug-perturbation-model/save/dev_perturb_kaggle_data-Jan17-09-51', type=str, help='Forecast result one')
    parser.add_argument('--label1', default='drug model kaggle_data 5000', type=str, help='Forecast label one')
    parser.add_argument('--savedir2', default="/home/zqliu02/code/gene-drug-perturbation-model/save/dev_perturb_kaggle_data_1200-Jan17-10-28", type=str, help='Forecast result two or None')
    parser.add_argument('--label2', default="drug model kaggle_data 1200", type=str, help='Forecast label two or None')
    parser.add_argument('--evaluate', default='Pearsonde20_delate', type=str, help='Evaluation index')
    parser.add_argument('--splits', nargs='+', default="/home/zqliu02/code/gene-drug-perturbation-model/data/kaggle_data_1200_bin/splits/kaggle_data_1200_bin_simulation_1_0.75.pkl", help='Training set and test set split file')
    args = parser.parse_args()
    with open(args.splits, 'rb') as file:
    # 读取文件内容
        splits_data = pickle.load(file)


    
    with open(f"{args.savedir1}/test_metrics_Pearson.json", "r") as json_file:
        gene1pred = json.load(json_file)
    PearsonDE20 = []
    for pert in splits_data["test"]:
        if pert in gene1pred.keys():
            PearsonDE20.append(gene1pred[pert][args.evaluate])
    


    plt.figure(figsize=(8, 6))
    sns.kdeplot(PearsonDE20, fill=True, color="blue", label=args.label1)
    if args.savedir2 is not None:
        with open(f"{args.savedir2}/test_metrics_Pearson.json", "r") as json_file:
            gene2pred = json.load(json_file)
        Pearson2DE20 = []
        for pert in splits_data["test"]:
            if pert in gene2pred.keys():
                Pearson2DE20.append(gene2pred[pert][args.evaluate])
        sns.kdeplot(Pearson2DE20, fill=True, color="green", label=args.label2)


    plt.title(args.evaluate)
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend(loc='upper left')
    plt.savefig(f"{args.savedir1}/{args.evaluate}.png", bbox_inches='tight')