from torch_geometric.data import Data
import torch
import numpy as np
import pickle
from torch_geometric.data import DataLoader
import os
import scanpy as sc
import networkx as nx
from tqdm import tqdm
import pandas as pd
import warnings
import random
from scipy import stats
warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

from .data_utils import get_DE_genes, get_dropout_non_zero_genes, DataSplitter,get_dropout_non_zero_genes
from .utils import print_sys, zip_data_download_wrapper, dataverse_download,\
                  filter_pert_in_go, get_genes_from_perts

class PertData:
    
    def __init__(self, data_path, perturb_way,
                 gene_set_path=None, 
                 default_pert_graph=True):
        
        # Dataset/Dataloader attributes
        self.data_path = data_path
        self.default_pert_graph = default_pert_graph
        self.gene_set_path = gene_set_path
        self.dataset_name = None
        self.dataset_path = None
        self.adata = None
        self.dataset_processed = None
        self.ctrl_adata = None
        self.gene_names = []
        self.node_map = {}
        self.perturb_way =perturb_way
        # Split attributes
        self.split = None
        self.seed = None
        self.subgroup = None
        self.train_gene_set_size = None

        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        server_path = 'https://dataverse.harvard.edu/api/access/datafile/6153417'
        dataverse_download(server_path,
                           os.path.join(self.data_path, 'gene2go_all.pkl'))
        with open(os.path.join(self.data_path, 'gene2go_all.pkl'), 'rb') as f:
            self.gene2go = pickle.load(f)
   
    def set_pert_genes(self):
        """
        Set the list of genes that can be perturbed and are to be included in 
        perturbation graph
        """
        
        if self.gene_set_path is not None:
            # If gene set specified for perturbation graph, use that
            path_ = self.gene_set_path
            self.default_pert_graph = False
            with open(path_, 'rb') as f:
                essential_genes = pickle.load(f)
            
        elif self.default_pert_graph is False:
            # Use a smaller perturbation graph 
            all_pert_genes = get_genes_from_perts(self.adata.obs['condition'])
            essential_genes = list(self.adata.var['gene_name'].values)
            essential_genes += all_pert_genes
            
        else:
            # Otherwise, use a large set of genes to create perturbation graph
            server_path = 'https://dataverse.harvard.edu/api/access/datafile/6934320'
            path_ = os.path.join(self.data_path,
                                     'essential_all_data_pert_genes.pkl')
            dataverse_download(server_path, path_)
            with open(path_, 'rb') as f:
                essential_genes = pickle.load(f)
        #gene2go = {i: self.gene2go[i] for i in essential_genes if i in self.gene2go}
        #all
        self.pert_names = np.unique(list(self.gene2go.keys()))
        

        self.node_map_pert = {x: it for it, x in enumerate(self.pert_names)}
        pert_names_dict = {name: index for index, name in enumerate(self.pert_names)}
        
# 初始化 self.gene_pert_idx
        self.gene_pert_idx = []
# 对于 gene_name 中的每个元素，查找其在 pert_names_dict 中的位置
        for i, name in enumerate(self.adata.var.gene_name):
            if name in pert_names_dict:
                self.gene_pert_idx.append([i, pert_names_dict[name]])

        #sc.pp.highly_variable_genes(self.adata, n_top_genes=1536,flavor="seurat_v3")
        #highly_variable_gene_indices = [i for i, x in enumerate(self.adata.var['highly_variable']) if x]
        #all_pert_genes = get_genes_from_perts(self.adata.obs['condition'])
        #highly_variable_pert_indices = [i for i, gene_name in enumerate(self.adata.var.gene_name) if gene_name in all_pert_genes]
        #highly_variable_gene_indices.extend(highly_variable_pert_indices)
        #unique_sorted_list = sorted(list(set(highly_variable_gene_indices)))

        #self.highly_variable_gene_indices=unique_sorted_list
        

          
    def load(self, data_name = None, data_path = None):
        """
        Load existing dataloader
        Use data_name for loading 'norman', 'adamson', 'dixit' datasets
        For other datasets use data_path
        """
        
        if data_name in ['norman', 'adamson', 'dixit']:
            ## load from harvard dataverse
            if data_name == 'norman':
                url = 'https://dataverse.harvard.edu/api/access/datafile/6154020'
            elif data_name == 'adamson':
                url = 'https://dataverse.harvard.edu/api/access/datafile/6154417'
            elif data_name == 'dixit':
                url = 'https://dataverse.harvard.edu/api/access/datafile/6154416'
            data_path = os.path.join(self.data_path, data_name)
            zip_data_download_wrapper(url, data_path, self.data_path)            
            self.dataset_name = data_path.split('/')[-1]
            self.dataset_path = data_path
            adata_path = os.path.join(data_path, 'perturb_processed.h5ad')
            self.adata = sc.read_h5ad(adata_path)
            #self.adata = get_DE_genes(self.adata,skip_calc_de =False)
            self.adata = get_dropout_non_zero_genes(self.adata)
            newadata_path = os.path.join(data_path, 'perturb_processednew1.h5ad')
            self.adata.write(newadata_path) # type: ignore

        elif os.path.exists(data_path):
            adata_path = os.path.join(data_path, 'perturb_processednew.h5ad')
            self.adata = sc.read_h5ad(adata_path)
            self.adata = get_dropout_non_zero_genes(self.adata)
            newadata_path = os.path.join(data_path, 'perturb_processednew1.h5ad')
            self.adata.write(newadata_path) # type: ignore
            self.dataset_name = data_path.split('/')[-1]
            self.dataset_path = data_path
        else:
            raise ValueError("data attribute is either Norman/Adamson/Dixit "
                             "or a path to an h5ad file")
        
        self.set_pert_genes()
        #print(list(self.pert_names))
        print_sys('These perturbations are not in the GO graph and their '
                  'perturbation can thus not be predicted')
        not_in_go_pert = np.array(self.adata.obs[
                                  self.adata.obs.condition.apply(
                                  lambda x:not filter_pert_in_go(x,
                                        self.pert_names))].condition.unique())
        print_sys(not_in_go_pert)
        
        filter_go = self.adata.obs[self.adata.obs.condition.apply(
                              lambda x: filter_pert_in_go(x, self.pert_names))]
        self.adata = self.adata[filter_go.index.values, :]
        pyg_path = os.path.join(data_path, 'data_pyg')
        if not os.path.exists(pyg_path):
            os.mkdir(pyg_path)
        dataset_fname = os.path.join(pyg_path, 'cell_graphs.pkl')
                
        if os.path.isfile(dataset_fname):
            print_sys("Local copy of pyg dataset is detected. Loading...")
            self.dataset_processed = pickle.load(open(dataset_fname, "rb"))        
            print_sys("Done!")
        else:
            self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']
            self.gene_names = self.adata.var.gene_name
            
            
            print_sys("Creating pyg object for each cell in the data...")
            self.dataset_processed = self.create_dataset_file()
            print_sys("Saving new dataset pyg object at " + dataset_fname) 
            pickle.dump(self.dataset_processed, open(dataset_fname, "wb"))    
            print_sys("Done!")
        
    def new_data_process(self, dataset_name,
                         adata = None,
                         skip_calc_de =False,
                         model_type=None):
        self.model_type = model_type
        if 'condition' not in adata.obs.columns.values:
            raise ValueError("Please specify condition")
        if 'gene_name' not in adata.var.columns.values:
            raise ValueError("Please specify gene name")
        if 'cell_type' not in adata.obs.columns.values:
            raise ValueError("Please specify cell type")
        
        dataset_name = dataset_name.lower()
        self.dataset_name = dataset_name
        save_data_folder = os.path.join(self.data_path, dataset_name)
        
        if not os.path.exists(save_data_folder):
            os.mkdir(save_data_folder)
        self.dataset_path = save_data_folder
        self.adata = get_DE_genes(adata, skip_calc_de,self.model_type)
        if not skip_calc_de:
            self.adata = get_dropout_non_zero_genes(self.adata)

        #self.perturb_screen(self.perturb_way)
        self.adata.write_h5ad(os.path.join(save_data_folder, 'perturb_processednew.h5ad'))
        if model_type =="gene":
            self.set_pert_genes()

            print_sys('These perturbations are not in the GO graph and their '
                    'perturbation can thus not be predicted')
            not_in_go_pert = np.array(self.adata.obs[
                                    self.adata.obs.condition.apply(
                                    lambda x:not filter_pert_in_go(x,
                                            self.pert_names))].condition.unique())
            print_sys(not_in_go_pert)
            
            filter_go = self.adata.obs[self.adata.obs.condition.apply(
                                lambda x: filter_pert_in_go(x, self.pert_names))]
            self.adata = self.adata[filter_go.index.values, :]
        elif model_type =="drug":
            print_sys('all drug perturbations are predictable')

        self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']

        self.cell_type_dict ={}
        for key in set(self.adata.obs["cell_type"]):
            self.cell_type_dict[key]=self.adata[(self.adata.obs["cell_type"]==key) & (self.adata.obs.condition == "ctrl")].to_df().mean().values

        self.gene_names = self.adata.var.gene_name
        pyg_path = os.path.join(save_data_folder, 'data_pyg')
        if not os.path.exists(pyg_path):
            os.mkdir(pyg_path)
        dataset_fname = os.path.join(pyg_path, 'cell_graphs.pkl')
        if os.path.isfile(dataset_fname):
            print_sys("Local copy of pyg dataset is detected. Loading...")
            self.dataset_processed = pickle.load(open(dataset_fname, "rb"))        
            print_sys("Done!")
        else:
            print_sys("Creating pyg object for each cell in the data...")
            if model_type =="gene":
                self.dataset_processed = self.create_dataset_file()
            elif model_type =="drug":
                self.dataset_processed = self.create_drug_dataset_file()
            print_sys("Saving new dataset pyg object at " + dataset_fname) 
            pickle.dump(self.dataset_processed, open(dataset_fname, "wb"))    
            print_sys("Done!")
        
    def prepare_split(self, split = 'simulation', 
                      seed = 1, 
                      train_gene_set_size = 0.75,
                      combo_seen2_train_frac = 0.75,
                      combo_single_split_test_set_fraction = 0.1,
                      test_perts = None,
                      only_test_set_perts = False,
                      test_pert_genes = None):
        available_splits = ['simulation', 'simulation_single', 'combo_seen0',
                            'combo_seen1', 'combo_seen2', 'single', 'no_test',
                            'no_split']
        if split not in available_splits:
            raise ValueError('currently, we only support ' + ','.join(available_splits))
        self.split = split
        self.seed = seed
        self.subgroup = None
        self.train_gene_set_size = train_gene_set_size
        
        split_folder = os.path.join(self.dataset_path, 'splits')
        if not os.path.exists(split_folder):
            os.mkdir(split_folder)
        split_file = self.dataset_name + '_' + split + '_' + str(seed) + '_' \
                                       +  str(train_gene_set_size) + '.pkl'
        split_path = os.path.join(split_folder, split_file)
        
        if test_perts:
            split_path = split_path[:-4] + '_' + test_perts + '.pkl'
        
        if os.path.exists(split_path):
            print_sys("Local copy of split is detected. Loading...")
            set2conditions = pickle.load(open(split_path, "rb"))
            if split == 'simulation':
                subgroup_path = split_path[:-4] + '_subgroup.pkl'
                subgroup = pickle.load(open(subgroup_path, "rb"))
                self.subgroup = subgroup
        else:
            print_sys("Creating new splits....")
            if test_perts:
                test_perts = test_perts.split('_')
                    
            if split in ['simulation', 'simulation_single']:
                DS = DataSplitter(self.adata, split_type=split)
                
                adata, subgroup = DS.split_data(train_gene_set_size = train_gene_set_size, 
                                                combo_seen2_train_frac = combo_seen2_train_frac,
                                                seed=seed,
                                                test_perts = test_perts,
                                                only_test_set_perts = only_test_set_perts
                                               )
                subgroup_path = split_path[:-4] + '_subgroup.pkl'
                pickle.dump(subgroup, open(subgroup_path, "wb"))
                self.subgroup = subgroup
                
            elif split[:5] == 'combo':
                split_type = 'combo'
                seen = int(split[-1])

                if test_pert_genes:
                    test_pert_genes = test_pert_genes.split('_')
                
                DS = DataSplitter(self.adata, split_type=split_type, seen=int(seen))
                adata = DS.split_data(test_size=combo_single_split_test_set_fraction,
                                      test_perts=test_perts,
                                      test_pert_genes=test_pert_genes,
                                      seed=seed)
            
            elif split == 'single':
                DS = DataSplitter(self.adata, split_type=split)
                adata = DS.split_data(test_size=combo_single_split_test_set_fraction,
                                      seed=seed)
            
            elif split == 'no_test':
                DS = DataSplitter(self.adata, split_type=split)
                adata = DS.split_data(test_size=combo_single_split_test_set_fraction,
                                      seed=seed)
            
            elif split == 'no_split':          
                adata = self.adata
                adata.obs['split'] = 'test'
            
            set2conditions = dict(adata.obs.groupby('split').agg({'condition':
                                                        lambda x: x}).condition)
            set2conditions = {i: j.unique().tolist() for i,j in set2conditions.items()} 
            pickle.dump(set2conditions, open(split_path, "wb"))
            print_sys("Saving new splits at " + split_path)
            
        self.set2conditions = set2conditions

        if split == 'simulation':
            print_sys('Simulation split test composition:')
            for i,j in subgroup['test_subgroup'].items():
                print_sys(i + ':' + str(len(j)))
        print_sys("Done!")
        
    def get_dataloader(self, batch_size, test_batch_size = None):
        if test_batch_size is None:
            test_batch_size = batch_size
            
        self.node_map = {x: it for it, x in enumerate(self.adata.var.gene_name)}
        self.gene_names = self.adata.var.gene_name
       
        # Create cell graphs
        cell_graphs = {}
        if self.split == 'no_split':
            i = 'test'
            cell_graphs[i] = []
            for p in self.set2conditions[i]:
                if p != 'ctrl':
                    cell_graphs[i].extend(self.dataset_processed[p])
                
            print_sys("Creating dataloaders....")
            # Set up dataloaders
            test_loader = DataLoader(cell_graphs['test'],
                                batch_size=batch_size, shuffle=False)

            print_sys("Dataloaders created...")
            return {'test_loader': test_loader}
        else:
            if self.split =='no_test':
                splits = ['train','val']
            else:
                splits = ['train','val','test']
            for i in splits:
                cell_graphs[i] = []
                for p in self.set2conditions[i]:
                    cell_graphs[i].extend(self.dataset_processed[p])

            print_sys("Creating dataloaders....")
            
            # Set up dataloaders
            train_loader = DataLoader(cell_graphs['train'],
                                batch_size=batch_size, shuffle=True, drop_last = True)
            val_loader = DataLoader(cell_graphs['val'],
                                batch_size=batch_size, shuffle=True)
            
            if self.split !='no_test':
                test_loader = DataLoader(cell_graphs['test'],
                                batch_size=batch_size, shuffle=False)
                self.dataloader =  {'train_loader': train_loader,
                                    'val_loader': val_loader,
                                    'test_loader': test_loader}

            else: 
                self.dataloader =  {'train_loader': train_loader,
                                    'val_loader': val_loader}
            print_sys("Done!")
        #del self.dataset_processed # clean up some memory
    
        
    def create_dataset_file(self):
        #gene_id2idx = dict(zip(adata.var.index.values, range(len(adata.var))))
        dl = {}
        for p in tqdm(self.adata.obs['condition'].unique()):
            cell_graph_dataset = self.create_cell_graph_dataset(self.adata, p, num_samples=1)
            dl[p] = cell_graph_dataset
        return dl
    
    def create_drug_dataset_file(self):
        #gene_id2idx = dict(zip(adata.var.index.values, range(len(adata.var))))
        dl = {}
        for p in tqdm(self.adata.obs['condition'].unique()):
            cell_graph_dataset = self.create_drug_cell_graph_dataset(self.adata, p, num_samples=1)
            dl[p] = cell_graph_dataset
        return dl
    
    def get_pert_idx(self, pert_category, adata_):
        try:
            pert_idx = [np.where(p == self.pert_names)[0][0]
                    for p in pert_category.split('+')
                    if p != 'ctrl']
        except:
            print(pert_category)
            pert_idx = None
            
        return pert_idx

    # Set up feature matrix and output
        
    def create_cell_graph(self, X, y, de_idx, de_idx20, de_idx50, de_idx100, de_idx200, de_idxall,pert_flags,pert,pert_idx=None,drug_pert = None):

        #pert_feats = np.expand_dims(pert_feats, 0)
        #feature_mat = torch.Tensor(np.concatenate([X, pert_feats])).T
        feature_mat = torch.Tensor(X).T
        
        '''
        pert_feats = np.zeros(len(self.pert_names))
        if pert_idx is not None:
            for p in pert_idx:
                pert_feats[int(np.abs(p))] = 1
        pert_feats = torch.Tensor(pert_feats).T
        '''
        if pert_idx is None:
            pert_idx = [-1]
        return Data(x=feature_mat, pert_idx=pert_idx,
                    y=torch.Tensor(y), de_idx=de_idx, de_idx20=de_idx20, de_idx50=de_idx50, de_idx100=de_idx100, de_idx200=de_idx200, de_idxall=de_idxall,pert_flags = pert_flags,pert=pert,drug_pert=drug_pert)

    def create_cell_graph_dataset(self, split_adata, pert_category,
                                  num_samples=1):
        """
        Combine cell graphs to create a dataset of cell graphs
        """
        gene_id2idx = dict(zip(split_adata.var.index.values, range(len(split_adata.var))))
        num_de_genes = 20        
        adata_ = split_adata[split_adata.obs['condition'] == pert_category]
        if 'rank_genes_groups_cov_all' in adata_.uns:
            de_genes = adata_.uns['rank_genes_groups_cov_all']
            de = True
        else:
            de = False
            num_de_genes = 1
        Xs = []
        ys = []
        pert_flags = []
            
      
        # When considering a non-control perturbation
        if pert_category != 'ctrl':
            # Get the indices of applied perturbation
            
            pert_idx = self.get_pert_idx(pert_category, adata_)



            # Store list of genes that are most differentially expressed for testing
            pert_de_category = adata_.obs['condition_name'][0]
            pert_flags = np.zeros(len(split_adata.var["gene_name"].tolist()))
            
            for i in range(0,len(self.gene_pert_idx)):
                if len(pert_idx) == 1:
                    if self.gene_pert_idx[i][1] == pert_idx[0]:
                        pert_flags[self.gene_pert_idx[i][0]]=1
                        
                if len(pert_idx) == 2:
                    if self.gene_pert_idx[i][1] == pert_idx[0]:
                        pert_flags[self.gene_pert_idx[i][0]]=1


                    if self.gene_pert_idx[i][1] == pert_idx[1]:
                        pert_flags[self.gene_pert_idx[i][0]]=1

        
       
                

        
            if de:
                de_idx = np.where(adata_.var_names.isin(
                np.array(de_genes[pert_de_category][:num_de_genes])))[0]
                #top 20 非零
                #.X.toarray
                try:
                    X = adata_.X.toarray()
                except:
                    X = adata_.X

                non_zero_cols = np.where(np.count_nonzero(X, axis=0) >= (X.shape[0] / 10))[0]
                non_zero = non_zero_cols
                depert = de_genes[pert_de_category]
                gene_idx_top = [gene_id2idx[i] for i in depert]
                non_zero_20 = [i for i in gene_idx_top if i in non_zero][:20]
                de_idx20 = np.array(non_zero_20)
                non_zero_50 = [i for i in gene_idx_top if i in non_zero][:50]
                de_idx50 = np.array(non_zero_50)

                non_zero_100 = [i for i in gene_idx_top if i in non_zero][:100]
                de_idx100 = np.array(non_zero_100)

                non_zero_200 = [i for i in gene_idx_top if i in non_zero][:200]
                de_idx200 = np.array(non_zero_200)

                non_zero_all = [i for i in gene_idx_top if i in non_zero]
                de_idxall = np.array(non_zero_all)
                

      
            else:
                de_idx = [-1] * num_de_genes
                de_idx20 = [-1] * 20
                de_idx50 = [-1] * 50
                de_idx100 = [-1] * 100
                de_idx200 = [-1] * 200
                de_idxall = [-1] * 500
            for i in range(adata_.shape[0]):
                # Use samples from control as basal expression
                Xs.append(self.cell_type_dict[adata_[i].obs["cell_type"].iloc[0]])
                ys.append(adata_[i].X[0])

        # When considering a control perturbation
        else:
            pert_idx = None
            pert_flags = np.full((len(split_adata.var["gene_name"].tolist()),), 2)
            de_idx = [-1] * num_de_genes
            de_idx20 = [-1] * 20
            de_idx50 = [-1] * 50
            de_idx100 = [-1] * 100
            de_idx200 = [-1] * 200
            de_idxall = [-1] * 500
            # for cell_z in adata_.X:
            #     Xs.append(cell_z)
            #     ys.append(cell_z)
            for i in range(adata_.shape[0]):
                Xs.append(self.cell_type_dict[adata_[i].obs["cell_type"].iloc[0]])
                ys.append(adata_[i].X[0])

        # Create cell graphs
        cell_graphs = []
        for X, y in zip(Xs, ys):
            try:
                cell_graphs.append(self.create_cell_graph(X,
                                    y.toarray(), de_idx, de_idx20, de_idx50, de_idx100, de_idx200, de_idxall,pert_flags, pert_category, pert_idx))
            except:
                cell_graphs.append(self.create_cell_graph(X,
                                    y, de_idx, de_idx20, de_idx50, de_idx100, de_idx200, de_idxall,pert_flags, pert_category, pert_idx))
        
        

        return cell_graphs
    

    def create_drug_cell_graph_dataset(self, split_adata, pert_category,
                                  num_samples=1):
        """
        Combine cell graphs to create a dataset of cell graphs
        """
        gene_id2idx = dict(zip(split_adata.var.index.values, range(len(split_adata.var))))
        num_de_genes = 20        
        adata_ = split_adata[split_adata.obs['condition'] == pert_category]
        if 'rank_genes_groups_cov_all' in adata_.uns:
            de_genes = adata_.uns['rank_genes_groups_cov_all']
            de = True
        else:
            de = False
            num_de_genes = 1
        Xs = []
        ys = []
        pert_flags = []
            
      
        # When considering a non-control perturbation
        if pert_category != 'ctrl':
            # Get the indices of applied perturbation
      
            # pert_idx = self.get_pert_idx(pert_category, adata_)
            
            if pert_category.rpartition('+')[0]=="ctrl":
                pert = pert_category.rpartition('+')[2]
            elif pert_category.rpartition('+')[2]=="ctrl":
                pert = pert_category.rpartition('+')[0]
            # Store list of genes that are most differentially expressed for testing
            pert_de_category = adata_.obs['condition_name'][0]
        

        
            if de:
                de_idx = np.where(adata_.var_names.isin(
                np.array(de_genes[pert_de_category][:num_de_genes])))[0]
                #top 20 非零
                #.X.toarray
                try:
                    X = adata_.X.toarray()
                except:
                    X = adata_.X

                non_zero_cols = np.where(np.count_nonzero(X, axis=0) >= (X.shape[0] / 10))[0]
                non_zero = non_zero_cols
                depert = de_genes[pert_de_category]
                gene_idx_top = [gene_id2idx[i] for i in depert]
                non_zero_20 = [i for i in gene_idx_top if i in non_zero][:20]
                de_idx20 = np.array(non_zero_20)
                non_zero_50 = [i for i in gene_idx_top if i in non_zero][:50]
                de_idx50 = np.array(non_zero_50)

                non_zero_100 = [i for i in gene_idx_top if i in non_zero][:100]
                de_idx100 = np.array(non_zero_100)

                non_zero_200 = [i for i in gene_idx_top if i in non_zero][:200]
                de_idx200 = np.array(non_zero_200)

                non_zero_all = [i for i in gene_idx_top if i in non_zero]
                de_idxall = np.array(non_zero_all)
                

      
            else:
                de_idx = [-1] * num_de_genes
                de_idx20 = [-1] * 20
                de_idx50 = [-1] * 50
                de_idx100 = [-1] * 100
                de_idx200 = [-1] * 200
                de_idxall = [-1] * 500
            # for cell_z in adata_.X:
            #     # Use samples from control as basal expression
            #     ctrl_samples = self.ctrl_adata[np.random.randint(0,
            #                             len(self.ctrl_adata), num_samples), :]
            #     for c in ctrl_samples.X:
            #         Xs.append(c)
            #         ys.append(cell_z)
            for i in range(adata_.shape[0]):
            # Use samples from control as basal expression
             
                Xs.append(self.cell_type_dict[adata_[i].obs["cell_type"].iloc[0]])
                
                ys.append(adata_[i].X[0])

        # When considering a control perturbation
        else:
            pert = "ctrl"
            de_idx = [-1] * num_de_genes
            de_idx20 = [-1] * 20
            de_idx50 = [-1] * 50
            de_idx100 = [-1] * 100
            de_idx200 = [-1] * 200
            de_idxall = [-1] * 500
            # for cell_z in adata_.X:
            #     Xs.append(cell_z)
            #     ys.append(cell_z)
            for i in range(adata_.shape[0]):
                # Use samples from control as basal expression
                Xs.append(self.cell_type_dict[adata_[i].obs["cell_type"].iloc[0]])
                ys.append(adata_[i].X[0])
     
        # Create cell graphs
        pert_flags =None
        pert_idx = None
        cell_graphs = []
        for X, y in zip(Xs, ys):
            try:
                cell_graphs.append(self.create_cell_graph(X,
                                    y.toarray(), de_idx, de_idx20, de_idx50, de_idx100, de_idx200, de_idxall,pert_flags, pert_category, pert_idx,drug_pert=pert))
            except:
                cell_graphs.append(self.create_cell_graph(X,
                                    y, de_idx, de_idx20, de_idx50, de_idx100, de_idx200, de_idxall,pert_flags, pert_category, pert_idx,drug_pert=pert))
            
        return cell_graphs

