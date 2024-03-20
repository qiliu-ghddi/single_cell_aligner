# Gene and Drug perturbation fine tuning model 
This code is for fine tuning Gene and Drug perturbation model 

2024 01.25 add Fine-tuning and evaluation of multiple cell types

#### Quick start
* run /home/cliang02/work/bin/cre-python data_preprocessing.py  
* run CUDA_VISIBLE_DEVICES=0 /home/cliang02/work/bin/cre-python  Perturbation-GPU-pred.py
* run /home/cliang02/work/bin/cre-python analysis/jsontoh5ad.py
* run /home/cliang02/work/bin/cre-python analysis/all_evaluation.py
* run /home/cliang02/work/bin/cre-python analysis/evaluate_perturbRes_prediction.py
* run CUDA_VISIBLE_DEVICES=0 /home/cliang02/work/bin/cre-python  Perturbation-GPU-eval.py


#### Folder

* scgpt     : Model code of fine tuning.  
* gears     : Data processing code.
* data      : Data required by fine tuning.
* analysis  : Evaluation index and visualization of the model.

#### data_preprocessing.py 
Required parameter



* adata_file     Input file path
* save           Output file path and name
* conditionname  Perturbation tags in the obs column in the file
* ctrlname       ctrl tags in the obs column in the file
* cell_type      cell type tags in the obs column in the file
* gene_name      gene_name in the var row in the file
* model_type     gene or drug


#### Perturbation-GPU-pred.py
Required parameter
    


* model_type           Choose whether to use the drug model or the gene model
* max_seq_len          Maximum sequence length for training
* load_model           Position of pre-trained model weights
* load_param_prefixs   Read a portion of the pre-trained model weights
* batch_size           Batch size for training
* eval_batch_size      Batch size for evaluation
* epochs               Number of epochs
* use_fast_transformer Whether to use flash attention
* data_name            Name of the dataset
* perts_to_plot        Perturbations to plot
* screen_list          perturbation without training and prediction
* adata_file           Path to adata file

#### jsontoh5ad.py
Required parameter

* adata_file     Input dataset perturb_processednew.h5ad file path
* savedir        Location where model prediction results are saved
* splits         The location of the training set and test set split file

#### all_evaluation.py
Required parameter

* input_h5ad     The input h5ad file recording expression profile of perturbed cells in prediction, required
* adata_file     Input dataset perturb_processednew.h5ad file path
* savedir        Location where model prediction results are saved
* splits         The location of the training set and test set split file

#### evaluate_perturbRes_prediction.py
Required parameter

* input_h5ad     The input h5ad file recording expression profile of perturbed cells in prediction, required 
* dataset_name   The dataset name involved in this analysis, used to construct the output file name, required
* output_dir     The output directory, required
* model_type     gene or drug

#### Perturbation-GPU-eval.py
Required parameter

* model_type           Choose whether to use the drug model or the gene model
* max_seq_len          Maximum sequence length for training
* load_model           Position of pre-trained model weights
* batch_size           Batch size for training
* eval_batch_size      Batch size for evaluation
* use_fast_transformer Whether to use flash attention
* perts_to_plot        Perturbations to plot and pred
* screen_list          perturbation without training and prediction
* adata_file           Path to adata file
