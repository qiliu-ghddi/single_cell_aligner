import json
import argparse
import csv

if __name__ == "__main__":

    # settings for data prcocessing
    parser = argparse.ArgumentParser(description='write results.')
    parser.add_argument('--savedir', default='/home/zqliu02/code/gene-drug-perturbation-model/save/dev_perturb_kaggle_data_1200_bin-Jan17-15-03', type=str, help='Path to forecast result file')
    args = parser.parse_args()
    #scGPT adamson
   
    metricslist = ["_MSE","_normalizeMSE","_Zscore","_direction","_Pearson","_pertvariation","_STD"]
    print(args.savedir)
    result = []
    for metrics in metricslist:

        with open(f"{args.savedir}/test_metrics{metrics}.json", "r") as json_file:
            gene1pred = json.load(json_file)
        if metrics =='_direction':
            
            result.append(["direction20",round(float(gene1pred["all"]['direction20']),4)])
            result.append(["direction100",round(float(gene1pred["all"]['direction100']),4)])
            result.append(["direction200",round(float(gene1pred["all"]['direction200']),4)])
            result.append(["directiondeall",round(float(gene1pred["all"]['directiondeall']),4)])
            result.append(["directionall",round(float(gene1pred["all"]['directionall']),4)])


        if metrics =='_normalizeMSE':
        
            result.append(["normalizeMSE20",round(float(gene1pred["20"]),4)])
            result.append(["normalizeMSE100",round(float(gene1pred["100"]),4)])
            result.append(["normalizeMSE200",round(float(gene1pred["200"]),4)])
            result.append(["normalizeMSEdeall",round(float(gene1pred["deall"]),4)])
            result.append(["normalizeMSEall",round(float(gene1pred["all"]),4)])

        if metrics =='_pertvariation':
        
            result.append(["pert_variation20",round(float(gene1pred["all"]['pert_variation20']),4)])
            result.append(["pert_variation100",round(float(gene1pred["all"]['pert_variation100']),4)])
            result.append(["pert_variation200",round(float(gene1pred["all"]['pert_variation200']),4)])
            result.append(["pert_variationdeall",round(float(gene1pred["all"]['pert_variationdeall']),4)])
            result.append(["pert_variationall",round(float(gene1pred["all"]['pert_variationall']),4)])
           

        if metrics =='_STD':

            result.append(["STD20",round(float(gene1pred["all"]['STD20']),4)])
            result.append(["STD100",round(float(gene1pred["all"]['STD100']),4)])
            result.append(["STD200",round(float(gene1pred["all"]['STD200']),4)])
            result.append(["STDdeall",round(float(gene1pred["all"]['STDdeall']),4)])

        
        if metrics =='_Zscore':
            result.append(["Zscore20",round(float(gene1pred["all"]['Zscore20']),4)])
            result.append(["Zscore100",round(float(gene1pred["all"]['Zscore100']),4)])
            result.append(["Zscore200",round(float(gene1pred["all"]['Zscore200']),4)])
    
        
        if metrics =='_Pearson':

            result.append(["Pearsonall",round(float(gene1pred["all"]["Pearsonall"]),4)])
            result.append(["Pearson20",round(float(gene1pred["all"]["Pearson20"]),4)])
            result.append(["Pearson100",round(float(gene1pred["all"]["Pearson100"]),4)])
            result.append(["Pearson200",round(float(gene1pred["all"]["Pearson200"]),4)])
            result.append(["Pearsondeall",round(float(gene1pred["all"]["Pearsondeall"]),4)])
            result.append(["Pearsondelate20",round(float(gene1pred["all"]["Pearson20_delate"]),4)])
            result.append(["Pearsondelate100",round(float(gene1pred["all"]["Pearson100_delate"]),4)])
            result.append(["Pearsondelate200",round(float(gene1pred["all"]["Pearson200_delate"]),4)])
            result.append(["Pearsondelatedeall",round(float(gene1pred["all"]["Pearsondeall_delate"]),4)])
     
            
        if metrics =='_MSE':

        
            result.append(["MSEall",round(float(gene1pred["all"]['MSEall']),4)])
            result.append(["MSE20",round(float(gene1pred["all"]['MSE20']),4)])
            result.append(["MSE100",round(float(gene1pred["all"]['MSE100']),4)])
            result.append(["MSE200",round(float(gene1pred["all"]['MSE200']),4)])
            result.append(["MSEdeall",round(float(gene1pred["all"]['MSEdeall']),4)])
    with open(args.savedir +  "/results_list.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(result)