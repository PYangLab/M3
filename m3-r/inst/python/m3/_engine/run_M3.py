import torch
import torch.nn as nn
from torch import exp, log
from torch import linalg as LA
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import os
import h5py
import scipy
import anndata
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from .util import setup_seed, MyDataset_mask, read_h5_data, read_fs_label, compute_zscore, plot_umaps, compute_log2, convert_to_longtensor, get_ref_query_data, KL_loss
from .util import load_and_merge_metadata, load_data_from_list, fill_missing_modalities, load_if_available, fill_and_concat_available_lists, process_count_matrix, process_ref_count, process_ref_count_after_imputation
from .util import create_new_generator, expand_encoder, expand_decoder, MyDataset_mask_train_query
from .model import M3_model, M3_model_wo_condition
from .train import train_M3, train_M3_with_query, train_M3_wo_condition
from .reclassify import ada_self_training


def create_full_data(modality1_path, modality2_path, modality3_path, metadata_path, save_path, condition_name, cty_name, batch_size, lr, num_epochs, min_delta, early_stop_patience, val_percentage, hvg_num, weight_modality, weight_batch_ae, embedding_dim, select_train_batch, select_test_batch):
    
    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.FloatTensor 
    LongTensor = torch.LongTensor 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    label, batch = load_and_merge_metadata(metadata_path)
    cty = convert_to_longtensor(label[cty_name])
    n_unique_cty = len(torch.unique(cty))
    condition = [convert_to_longtensor(label[c]) for c in condition_name]
    n_unique_conditions = [len(torch.unique(convert_to_longtensor(label[c]))) for c in condition_name]
    
    count_rna_list = load_if_available(modality1_path)
    count_adt_list = load_if_available(modality2_path)
    count_atac_list = load_if_available(modality3_path)
    count_rna, count_adt, count_atac = fill_and_concat_available_lists(count_rna_list, count_adt_list, count_atac_list)
    
    count_rna = process_count_matrix(count_rna, count_rna_list, hvg_num[0])
    count_adt = process_count_matrix(count_adt, count_adt_list, hvg_num[1])
    count_atac = process_count_matrix(count_atac, count_atac_list, hvg_num[2])
    
    ref_b, ref_c, ref_cty, ref_count_rna, ref_count_adt, ref_count_atac, ref_metadata = get_ref_query_data(batch, condition, cty, count_rna, count_adt, count_atac, label, select_batch=select_train_batch)
    
    ref_mask_poe_list = []
    ref_mask_recon_list = []
    ref_data_list = []
    if ref_count_rna is not None:
        ref_mask_poe_list, ref_mask_recon_list, ref_data_list, ref_mask_batch = process_ref_count(ref_count_rna, device, ref_mask_poe_list, ref_mask_recon_list, ref_data_list)
    if ref_count_adt is not None:
        ref_mask_poe_list, ref_mask_recon_list, ref_data_list, ref_mask_batch = process_ref_count(ref_count_adt, device, ref_mask_poe_list, ref_mask_recon_list, ref_data_list)
    if ref_count_atac is not None:
        ref_mask_poe_list, ref_mask_recon_list, ref_data_list, ref_mask_batch = process_ref_count(ref_count_atac, device, ref_mask_poe_list, ref_mask_recon_list, ref_data_list)
    
    # Final concatenation
    ref_data_list = [torch.nan_to_num(d, nan=0.0) for d in ref_data_list]
    ref_mask_poe_full = torch.cat(ref_mask_poe_list + [ref_mask_batch[:, 1].unsqueeze(1).to(device)], dim=1)
    ref_mask_recon_full = torch.cat(ref_mask_recon_list, dim=1)
    ref_data_full = torch.cat(ref_data_list, dim=1)
    return ref_data_full, ref_b, ref_mask_poe_full


def subsample_by_batch(ref_b, ref_c, ref_cty, ref_count_rna, ref_count_adt, ref_count_atac, ref_metadata):
    ref_b_np = np.array(ref_b)
    
    unique_batches, counts = np.unique(ref_b_np, return_counts=True)
    min_count = counts.min()
    print("Batch counts:", dict(zip(unique_batches, counts)))
    print("Minimum batch size:", min_count)
    
    keep_idx = []
    for b in unique_batches:
        idx = np.where(ref_b_np == b)[0]
        if len(idx) > min_count:
            sampled = np.random.choice(idx, min_count, replace=False)
        else:
            sampled = idx
        keep_idx.extend(sampled)
    keep_idx = np.random.permutation(keep_idx)
    
    ref_b_sub = torch.tensor(ref_b_np[keep_idx], dtype=torch.long)
    ref_c_sub = [torch.tensor(np.array(c)[keep_idx], dtype=torch.long) for c in ref_c]
    ref_cty_sub = torch.tensor(np.array(ref_cty)[keep_idx], dtype=torch.long)
    if ref_count_rna!=None:
        ref_count_rna_sub = ref_count_rna[keep_idx, :]
    else: 
        ref_count_rna_sub = None
    if ref_count_adt!=None:
        ref_count_adt_sub = ref_count_adt[keep_idx, :]
    else:
        ref_count_adt_sub = None
    if ref_count_atac!=None:
        ref_count_atac_sub = ref_count_atac[keep_idx, :]
    else:
        ref_count_atac_sub = None
    ref_metadata_sub = ref_metadata.iloc[keep_idx, :].reset_index(drop=True)
    return ref_b_sub, ref_c_sub, ref_cty_sub, ref_count_rna_sub, ref_count_adt_sub, ref_count_atac_sub, ref_metadata_sub

def subsample_by_batch_with_imputed_data(ref_b, ref_c, ref_cty, ref_count_rna, ref_count_adt, ref_count_atac, ref_count_imputed_rna, ref_count_imputed_adt, ref_count_imputed_atac, ref_metadata):
    ref_b_np = np.array(ref_b)
    
    unique_batches, counts = np.unique(ref_b_np, return_counts=True)
    min_count = counts.min()
    print("Batch counts:", dict(zip(unique_batches, counts)))
    print("Minimum batch size:", min_count)
    
    keep_idx = []
    for b in unique_batches:
        idx = np.where(ref_b_np == b)[0]
        if len(idx) > min_count:
            sampled = np.random.choice(idx, min_count, replace=False)
        else:
            sampled = idx
        keep_idx.extend(sampled)
    keep_idx = np.random.permutation(keep_idx)
    
    ref_b_sub = torch.tensor(ref_b_np[keep_idx], dtype=torch.long)
    ref_c_sub = [torch.tensor(np.array(c)[keep_idx], dtype=torch.long) for c in ref_c]
    ref_cty_sub = torch.tensor(np.array(ref_cty)[keep_idx], dtype=torch.long)
    if ref_count_rna!=None:
        ref_count_rna_sub = ref_count_rna[keep_idx, :]
    else: 
        ref_count_rna_sub = None
    if ref_count_adt!=None:
        ref_count_adt_sub = ref_count_adt[keep_idx, :]
    else:
        ref_count_adt_sub = None
    if ref_count_atac!=None:
        ref_count_atac_sub = ref_count_atac[keep_idx, :]
    else:
        ref_count_atac_sub = None

    if ref_count_imputed_rna!=None:
        ref_count_imputed_rna_sub = ref_count_imputed_rna[keep_idx, :]
    else: 
        ref_count_imputed_rna_sub = None
    if ref_count_imputed_adt!=None:
        ref_count_imputed_adt = ref_count_imputed_adt[keep_idx, :]
    else:
        ref_count_imputed_adt = None
    if ref_count_imputed_atac!=None:
        ref_count_imputed_atac = ref_count_imputed_atac[keep_idx, :]
    else:
        ref_count_imputed_atac = None
        
    ref_metadata_sub = ref_metadata.iloc[keep_idx, :].reset_index(drop=True)
    return ref_b_sub, ref_c_sub, ref_cty_sub, ref_count_rna_sub, ref_count_adt_sub, ref_count_atac_sub, ref_count_imputed_rna_sub, ref_count_imputed_adt_sub, ref_count_imputed_atac_sub, ref_metadata_sub


def run_M3(modality1_path, modality2_path, modality3_path, metadata_path, save_path, condition_name, cty_name, batch_size, lr, num_epochs, 
            min_delta, early_stop_patience, val_percentage, hvg_num, weight_modality, weight_batch_ae, embedding_dim, 
            select_train_batch, select_test_batch, balance_training=False):

    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.FloatTensor 
    LongTensor = torch.LongTensor 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label, batch = load_and_merge_metadata(metadata_path)
    cty = convert_to_longtensor(label[cty_name])
    n_unique_cty = len(torch.unique(cty))
    condition = [convert_to_longtensor(label[c]) for c in condition_name]
    n_unique_conditions = [len(torch.unique(convert_to_longtensor(label[c]))) for c in condition_name]
    
    count_rna_list = load_if_available(modality1_path)
    count_adt_list = load_if_available(modality2_path)
    count_atac_list = load_if_available(modality3_path)
    count_rna, count_adt, count_atac = fill_and_concat_available_lists(count_rna_list, count_adt_list, count_atac_list)
    
    count_rna = process_count_matrix(count_rna, count_rna_list, hvg_num[0])
    count_adt = process_count_matrix(count_adt, count_adt_list, hvg_num[1])
    count_atac = process_count_matrix(count_atac, count_atac_list, hvg_num[2])

    if balance_training == True:
        ref_b_full, ref_c_full, ref_cty_full, ref_count_rna_full, ref_count_adt_full, ref_count_atac_full, ref_metadata_full = get_ref_query_data(batch, condition, cty, count_rna, count_adt, count_atac, label, select_batch=select_train_batch)
        ref_b, ref_c, ref_cty, ref_count_rna, ref_count_adt, ref_count_atac, ref_metadata = subsample_by_batch(ref_b_full, ref_c_full, ref_cty_full, ref_count_rna_full, ref_count_adt_full, ref_count_atac_full, ref_metadata_full)
    else:
        ref_b, ref_c, ref_cty, ref_count_rna, ref_count_adt, ref_count_atac, ref_metadata = get_ref_query_data(batch, condition, cty, count_rna, count_adt, count_atac, label, select_batch=select_train_batch)
        
    ref_mask_poe_list = []
    ref_mask_recon_list = []
    ref_data_list = []
    if ref_count_rna is not None:
        ref_mask_poe_list, ref_mask_recon_list, ref_data_list, ref_mask_batch = process_ref_count(ref_count_rna, device, ref_mask_poe_list, ref_mask_recon_list, ref_data_list)
    if ref_count_adt is not None:
        ref_mask_poe_list, ref_mask_recon_list, ref_data_list, ref_mask_batch = process_ref_count(ref_count_adt, device, ref_mask_poe_list, ref_mask_recon_list, ref_data_list)
    if ref_count_atac is not None:
        ref_mask_poe_list, ref_mask_recon_list, ref_data_list, ref_mask_batch = process_ref_count(ref_count_atac, device, ref_mask_poe_list, ref_mask_recon_list, ref_data_list)
    
    # Final concatenation
    ref_data_list = [torch.nan_to_num(d, nan=0.0) for d in ref_data_list]
    ref_mask_poe = torch.cat(ref_mask_poe_list + [ref_mask_batch[:, 1].unsqueeze(1).to(device)], dim=1)
    ref_mask_recon = torch.cat(ref_mask_recon_list, dim=1)
    ref_data = torch.cat(ref_data_list, dim=1)
    transformed_dataset = MyDataset_mask(ref_data,  ref_mask_recon, ref_mask_poe, ref_cty, ref_b, ref_c)
    
    n_unique_ref_batch = len(torch.unique(ref_b))
    n_features = [d.shape[1] for d in ref_data_list] 
    classify_dim = torch.max(cty)+1
    
    if select_test_batch!=None:
        query_b, query_c, query_cty, query_count_rna, query_count_adt, query_count_atac, query_metadata = get_ref_query_data(batch, condition, cty, count_rna, count_adt, count_atac, label, select_batch=select_test_batch)
        query_b = query_b + max(ref_b) + 1
        query_mask_poe_list = []
        query_mask_recon_list = []
        query_data_list = []
        if query_count_rna is not None:
            query_mask_poe_list, query_mask_recon_list, query_data_list, query_mask_batch = process_ref_count(query_count_rna, device, query_mask_poe_list, query_mask_recon_list, query_data_list)
        if query_count_adt is not None:
            query_mask_poe_list, query_mask_recon_list, query_data_list, query_mask_batch = process_ref_count(query_count_adt, device, query_mask_poe_list, query_mask_recon_list, query_data_list)
        if query_count_atac is not None:
            query_mask_poe_list, query_mask_recon_list, query_data_list, query_mask_batch = process_ref_count(query_count_atac, device, query_mask_poe_list, query_mask_recon_list, query_data_list)
    
        n_unique_query_batch = len(torch.unique(query_b))
    
        # Final concatenation
        query_data_list = [torch.nan_to_num(d, nan=0.0) for d in query_data_list]
        query_mask_poe = torch.cat(query_mask_poe_list + [query_mask_batch[:, 1].unsqueeze(1).to(device)], dim=1)
        query_mask_recon = torch.cat(query_mask_recon_list, dim=1)
        query_data = torch.cat(query_data_list, dim=1)
    
        all_mask_poe_list = ref_mask_poe_list + query_mask_poe_list
        all_mask_recon_list = ref_mask_recon_list + query_mask_recon_list
        all_data_list = ref_data_list + query_data_list
        all_mask_batch = torch.cat([ref_mask_batch, query_mask_batch],0)
        
        all_data = torch.cat([ref_data, query_data], 0)
        all_batch = torch.cat([ref_b, query_b], 0)
        all_mask_recon =torch.cat([ref_mask_recon, query_mask_recon], 0)
        all_mask_poe =torch.cat([ref_mask_poe, query_mask_poe], 0)
        all_cty =torch.cat([ref_cty, query_cty], 0)
        all_b =torch.cat([ref_b, query_b], 0)
        all_c = [torch.cat([r.reshape(-1),
                            q.to(device=r.device, dtype=r.dtype).reshape(-1)], dim=0)
                 for r, q in zip(ref_c, query_c)]
        all_train_query_info = torch.cat([
            torch.ones(ref_data.shape[0], dtype=torch.long, device=device),  # ref -> 1
            torch.zeros(query_data.shape[0], dtype=torch.long, device=device) # query -> 0
        ], dim=0)
        all_transformed_dataset = MyDataset_mask_train_query(all_data, all_mask_recon, all_mask_poe, all_cty, all_b, all_c, all_train_query_info)
        n_unique_batch = n_unique_ref_batch + n_unique_query_batch
    
    else:
        all_data = ref_data
        all_mask_recon = ref_mask_recon
        all_mask_poe = ref_mask_poe
        all_cty = ref_cty
        all_b = ref_b
        all_c = ref_c
        all_transformed_dataset = MyDataset_mask(all_data, all_mask_recon, all_mask_poe, all_cty, all_b, all_c)
        n_unique_batch = n_unique_ref_batch
    
    total_len = len(all_transformed_dataset)
    val_len = int(val_percentage * total_len)
    all_len = total_len - val_len
    
    all_dataset, val_dataset = random_split(all_transformed_dataset, [all_len, val_len])
    all_dl = DataLoader(all_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    criterion_smooth_cty = nn.CrossEntropyLoss()
    criterion = nn.MSELoss().to(device)
    criterion_kl = KL_loss()
    
    generator = M3_model_wo_condition(n_features,
                                     batch_classify_dim=n_unique_batch, 
                                     condition_dim=n_unique_conditions, z_dim=embedding_dim, hidden_features=[embedding_dim, embedding_dim]).to(device)
    
    generator = train_M3_wo_condition(all_dl, val_dl, generator, criterion_smooth_cty, criterion, criterion_kl, device,
                                                       lr=lr, num_epochs=num_epochs, batch_classify_dim=n_unique_batch, condition_dim=n_unique_conditions,    min_delta=min_delta,
                                                       early_stop_patience = early_stop_patience, weight_batch_ae=weight_batch_ae, weight_modality=weight_modality, nfeatures=n_features)

    import copy
    generator_1stage = copy.deepcopy(generator)
    x, x_batch, z_embedding, z_batch, z_c, cla_cty1, cla_batch1, cla_c, mu, var = generator.cpu()(ref_data.to("cpu"), ref_b.to("cpu"), ref_mask_poe.to("cpu"))
    z_embedding = torch.where(torch.isnan(z_embedding) | torch.isinf(z_embedding), torch.tensor(0.0, device=z_embedding.device), z_embedding)
    
    preds = [t.clone() for t in ref_c]
    for i in range(len(ref_c)):
        new_data = z_embedding
        y_true = ref_c[i]
        num_classes = int(y_true.max().item() + 1)
        out = ada_self_training(
            X_tensor=new_data, y0_tensor=y_true, num_classes=num_classes,
            n_iters=5, seed_ratio=0.2, sample_ratio=0.2,
            alpha_start=1.6, alpha_end=1.0, tau=0.3, relabeled_weight=0.5,
            use_pca=True, pca_var=0.70, max_pcs=20, min_pcs=10,
            epochs_per_iter=30, batch_size=256, lr=1e-3, random_state=42,
            print_progress=True)
        preds[i] = out["final_pred"]
    
    ref_c = torch.as_tensor(preds, device=device)
    if select_test_batch!=None:
        query_b, query_c, query_cty, query_count_rna, query_count_adt, query_count_atac, query_metadata = get_ref_query_data(batch, condition, cty, count_rna, count_adt, count_atac, label, select_batch=select_test_batch)
        query_b = query_b + max(ref_b) + 1
        query_mask_poe_list = []
        query_mask_recon_list = []
        query_data_list = []
        if query_count_rna is not None:
            query_mask_poe_list, query_mask_recon_list, query_data_list, query_mask_batch = process_ref_count(query_count_rna, device, query_mask_poe_list, query_mask_recon_list, query_data_list)
        if query_count_adt is not None:
            query_mask_poe_list, query_mask_recon_list, query_data_list, query_mask_batch = process_ref_count(query_count_adt, device, query_mask_poe_list, query_mask_recon_list, query_data_list)
        if query_count_atac is not None:
            query_mask_poe_list, query_mask_recon_list, query_data_list, query_mask_batch = process_ref_count(query_count_atac, device, query_mask_poe_list, query_mask_recon_list, query_data_list)
    
        n_unique_query_batch = len(torch.unique(query_b))
    
        # Final concatenation
        query_data_list = [torch.nan_to_num(d, nan=0.0) for d in query_data_list]
        query_mask_poe = torch.cat(query_mask_poe_list + [query_mask_batch[:, 1].unsqueeze(1).to(device)], dim=1)
        query_mask_recon = torch.cat(query_mask_recon_list, dim=1)
        query_data = torch.cat(query_data_list, dim=1)
    
        all_mask_poe_list = ref_mask_poe_list + query_mask_poe_list
        all_mask_recon_list = ref_mask_recon_list + query_mask_recon_list
        all_data_list = ref_data_list + query_data_list
        all_mask_batch = torch.cat([ref_mask_batch, query_mask_batch],0)
        
        all_data = torch.cat([ref_data, query_data], 0)
        all_batch = torch.cat([ref_b, query_b], 0)
        all_mask_recon =torch.cat([ref_mask_recon, query_mask_recon], 0)
        all_mask_poe =torch.cat([ref_mask_poe, query_mask_poe], 0)
        all_cty =torch.cat([ref_cty, query_cty], 0)
        all_b =torch.cat([ref_b, query_b], 0)
        all_c = [torch.cat([r.reshape(-1),
                            q.to(device=r.device, dtype=r.dtype).reshape(-1)], dim=0)
                 for r, q in zip(ref_c, query_c)]
        all_train_query_info = torch.cat([
            torch.ones(ref_data.shape[0], dtype=torch.long, device=device),  # ref -> 1
            torch.zeros(query_data.shape[0], dtype=torch.long, device=device) # query -> 0
        ], dim=0)
        all_transformed_dataset = MyDataset_mask_train_query(all_data, all_mask_recon, all_mask_poe, all_cty, all_b, all_c, all_train_query_info)
        n_unique_batch = n_unique_ref_batch + n_unique_query_batch
    
    else:
        all_data = ref_data
        all_mask_recon = ref_mask_recon
        all_mask_poe = ref_mask_poe
        all_cty = ref_cty
        all_b = ref_b
        all_c = ref_c
        all_transformed_dataset = MyDataset_mask(all_data, all_mask_recon, all_mask_poe, all_cty, all_b, all_c)
        query_data = None
        query_b = None
        query_mask_poe = None
        query_metadata = None
        n_unique_batch = n_unique_ref_batch
    
    total_len = len(all_transformed_dataset)
    val_len = int(val_percentage * total_len)
    all_len = total_len - val_len
    
    all_dataset, val_dataset = random_split(all_transformed_dataset, [all_len, val_len])
    all_dl = DataLoader(all_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    generator = M3_model(n_features,
                                     batch_classify_dim=n_unique_batch, 
                                     condition_dim=n_unique_conditions, z_dim=embedding_dim, hidden_features=[embedding_dim, embedding_dim]).to(device)
    
    generator = train_M3(all_dl, val_dl, generator, criterion_smooth_cty, criterion, criterion_kl, device,
                                                       lr=lr, num_epochs=num_epochs, batch_classify_dim=n_unique_batch, condition_dim=n_unique_conditions,   min_delta = min_delta,
                                                       early_stop_patience = early_stop_patience, weight_batch_ae=weight_batch_ae, weight_modality=weight_modality, nfeatures=n_features)
    
    return ref_data, ref_b, ref_mask_poe, ref_metadata, query_data, query_b, query_mask_poe, query_metadata, generator, generator_1stage, preds[0], preds[1] 













def run_M3_update(modality1_path, modality2_path, modality3_path, metadata_path, save_path, condition_name, cty_name, batch_size, lr, num_epochs, 
            min_delta, early_stop_patience, val_percentage, hvg_num, weight_modality, weight_batch_ae, embedding_dim, 
            select_train_batch, select_test_batch, balance_training=False):

    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.FloatTensor 
    LongTensor = torch.LongTensor 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label, batch = load_and_merge_metadata(metadata_path)
    cty = convert_to_longtensor(label[cty_name])
    n_unique_cty = len(torch.unique(cty))
    condition = [convert_to_longtensor(label[c]) for c in condition_name]
    n_unique_conditions = [len(torch.unique(convert_to_longtensor(label[c]))) for c in condition_name]
    
    count_rna_list = load_if_available(modality1_path)
    count_adt_list = load_if_available(modality2_path)
    count_atac_list = load_if_available(modality3_path)
    count_rna, count_adt, count_atac = fill_and_concat_available_lists(count_rna_list, count_adt_list, count_atac_list)
    
    count_rna = process_count_matrix(count_rna, count_rna_list, hvg_num[0])
    count_adt = process_count_matrix(count_adt, count_adt_list, hvg_num[1])
    count_atac = process_count_matrix(count_atac, count_atac_list, hvg_num[2])

    if balance_training == True:
        ref_b_full, ref_c_full, ref_cty_full, ref_count_rna_full, ref_count_adt_full, ref_count_atac_full, ref_metadata_full = get_ref_query_data(batch, condition, cty, count_rna, count_adt, count_atac, label, select_batch=select_train_batch)
        ref_b, ref_c, ref_cty, ref_count_rna, ref_count_adt, ref_count_atac, ref_metadata = subsample_by_batch(ref_b_full, ref_c_full, ref_cty_full, ref_count_rna_full, ref_count_adt_full, ref_count_atac_full, ref_metadata_full)
    else:
        ref_b, ref_c, ref_cty, ref_count_rna, ref_count_adt, ref_count_atac, ref_metadata = get_ref_query_data(batch, condition, cty, count_rna, count_adt, count_atac, label, select_batch=select_train_batch)
        
    ref_mask_poe_list = []
    ref_mask_recon_list = []
    ref_data_list = []
    if ref_count_rna is not None:
        ref_mask_poe_list, ref_mask_recon_list, ref_data_list, ref_mask_batch = process_ref_count(ref_count_rna, device, ref_mask_poe_list, ref_mask_recon_list, ref_data_list)
    if ref_count_adt is not None:
        ref_mask_poe_list, ref_mask_recon_list, ref_data_list, ref_mask_batch = process_ref_count(ref_count_adt, device, ref_mask_poe_list, ref_mask_recon_list, ref_data_list)
    if ref_count_atac is not None:
        ref_mask_poe_list, ref_mask_recon_list, ref_data_list, ref_mask_batch = process_ref_count(ref_count_atac, device, ref_mask_poe_list, ref_mask_recon_list, ref_data_list)
    
    # Final concatenation
    ref_data_list = [torch.nan_to_num(d, nan=0.0) for d in ref_data_list]
    ref_mask_poe = torch.cat(ref_mask_poe_list + [ref_mask_batch[:, 1].unsqueeze(1).to(device)], dim=1)
    ref_mask_recon = torch.cat(ref_mask_recon_list, dim=1)
    ref_data = torch.cat(ref_data_list, dim=1)
    transformed_dataset = MyDataset_mask(ref_data,  ref_mask_recon, ref_mask_poe, ref_cty, ref_b, ref_c)
    
    n_unique_ref_batch = len(torch.unique(ref_b))
    n_features = [d.shape[1] for d in ref_data_list] 
    classify_dim = torch.max(cty)+1
    
    if select_test_batch!=None:
        query_b, query_c, query_cty, query_count_rna, query_count_adt, query_count_atac, query_metadata = get_ref_query_data(batch, condition, cty, count_rna, count_adt, count_atac, label, select_batch=select_test_batch)
        query_b = query_b + max(ref_b) + 1
        query_mask_poe_list = []
        query_mask_recon_list = []
        query_data_list = []
        if query_count_rna is not None:
            query_mask_poe_list, query_mask_recon_list, query_data_list, query_mask_batch = process_ref_count(query_count_rna, device, query_mask_poe_list, query_mask_recon_list, query_data_list)
        if query_count_adt is not None:
            query_mask_poe_list, query_mask_recon_list, query_data_list, query_mask_batch = process_ref_count(query_count_adt, device, query_mask_poe_list, query_mask_recon_list, query_data_list)
        if query_count_atac is not None:
            query_mask_poe_list, query_mask_recon_list, query_data_list, query_mask_batch = process_ref_count(query_count_atac, device, query_mask_poe_list, query_mask_recon_list, query_data_list)
    
        n_unique_query_batch = len(torch.unique(query_b))
    
        # Final concatenation
        query_data_list = [torch.nan_to_num(d, nan=0.0) for d in query_data_list]
        query_mask_poe = torch.cat(query_mask_poe_list + [query_mask_batch[:, 1].unsqueeze(1).to(device)], dim=1)
        query_mask_recon = torch.cat(query_mask_recon_list, dim=1)
        query_data = torch.cat(query_data_list, dim=1)
    
        all_mask_poe_list = ref_mask_poe_list + query_mask_poe_list
        all_mask_recon_list = ref_mask_recon_list + query_mask_recon_list
        all_data_list = ref_data_list + query_data_list
        all_mask_batch = torch.cat([ref_mask_batch, query_mask_batch],0)
        
        all_data = torch.cat([ref_data, query_data], 0)
        all_batch = torch.cat([ref_b, query_b], 0)
        all_mask_recon =torch.cat([ref_mask_recon, query_mask_recon], 0)
        all_mask_poe =torch.cat([ref_mask_poe, query_mask_poe], 0)
        all_cty =torch.cat([ref_cty, query_cty], 0)
        all_b =torch.cat([ref_b, query_b], 0)
        all_c = [torch.cat([r.reshape(-1),
                            q.to(device=r.device, dtype=r.dtype).reshape(-1)], dim=0)
                 for r, q in zip(ref_c, query_c)]
        all_train_query_info = torch.cat([
            torch.ones(ref_data.shape[0], dtype=torch.long, device=device),  # ref -> 1
            torch.zeros(query_data.shape[0], dtype=torch.long, device=device) # query -> 0
        ], dim=0)
        all_transformed_dataset = MyDataset_mask_train_query(all_data, all_mask_recon, all_mask_poe, all_cty, all_b, all_c, all_train_query_info)
        n_unique_batch = n_unique_ref_batch + n_unique_query_batch
    
    else:
        all_data = ref_data
        all_mask_recon = ref_mask_recon
        all_mask_poe = ref_mask_poe
        all_cty = ref_cty
        all_b = ref_b
        all_c = ref_c
        all_transformed_dataset = MyDataset_mask(all_data, all_mask_recon, all_mask_poe, all_cty, all_b, all_c)
        n_unique_batch = n_unique_ref_batch
    
    total_len = len(all_transformed_dataset)
    val_len = int(val_percentage * total_len)
    all_len = total_len - val_len
    
    all_dataset, val_dataset = random_split(all_transformed_dataset, [all_len, val_len])
    all_dl = DataLoader(all_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    criterion_smooth_cty = nn.CrossEntropyLoss()
    criterion = nn.MSELoss().to(device)
    criterion_kl = KL_loss()
    
    preds = [t.clone() for t in ref_c]
    for i in range(len(ref_c)):
        new_data = ref_data[:,0:1000]
        y_true = ref_c[i]
        num_classes = int(y_true.max().item() + 1)
        out = ada_self_training(
            X_tensor=new_data, y0_tensor=y_true, num_classes=num_classes,
            n_iters=5, seed_ratio=0.2, sample_ratio=0.2,
            alpha_start=1.6, alpha_end=1.0, tau=0.3, relabeled_weight=0.5,
            use_pca=True, pca_var=0.70, max_pcs=20, min_pcs=10,
            epochs_per_iter=30, batch_size=256, lr=1e-3, random_state=42,
            print_progress=True)
        preds[i] = out["final_pred"]
    
    ref_c = torch.as_tensor(preds, device=device)

    
    preds = [t.clone() for t in ref_c]
    for i in range(len(ref_c)):
        new_data = ref_data[:,0:1000]
        y_true = ref_c[i]
        num_classes = int(y_true.max().item() + 1)
        out = ada_self_training(
            X_tensor=new_data, y0_tensor=y_true, num_classes=num_classes,
            n_iters=5, seed_ratio=0.2, sample_ratio=0.2,
            alpha_start=1.6, alpha_end=1.0, tau=0.3, relabeled_weight=0.5,
            use_pca=True, pca_var=0.70, max_pcs=20, min_pcs=10,
            epochs_per_iter=30, batch_size=256, lr=1e-3, random_state=42,
            print_progress=True)
        preds[i] = out["final_pred"]
    
    ref_c = torch.as_tensor(preds, device=device)
    
    if select_test_batch!=None:
        query_b, query_c, query_cty, query_count_rna, query_count_adt, query_count_atac, query_metadata = get_ref_query_data(batch, condition, cty, count_rna, count_adt, count_atac, label, select_batch=select_test_batch)
        query_b = query_b + max(ref_b) + 1
        query_mask_poe_list = []
        query_mask_recon_list = []
        query_data_list = []
        if query_count_rna is not None:
            query_mask_poe_list, query_mask_recon_list, query_data_list, query_mask_batch = process_ref_count(query_count_rna, device, query_mask_poe_list, query_mask_recon_list, query_data_list)
        if query_count_adt is not None:
            query_mask_poe_list, query_mask_recon_list, query_data_list, query_mask_batch = process_ref_count(query_count_adt, device, query_mask_poe_list, query_mask_recon_list, query_data_list)
        if query_count_atac is not None:
            query_mask_poe_list, query_mask_recon_list, query_data_list, query_mask_batch = process_ref_count(query_count_atac, device, query_mask_poe_list, query_mask_recon_list, query_data_list)
    
        n_unique_query_batch = len(torch.unique(query_b))
    
        # Final concatenation
        query_data_list = [torch.nan_to_num(d, nan=0.0) for d in query_data_list]
        query_mask_poe = torch.cat(query_mask_poe_list + [query_mask_batch[:, 1].unsqueeze(1).to(device)], dim=1)
        query_mask_recon = torch.cat(query_mask_recon_list, dim=1)
        query_data = torch.cat(query_data_list, dim=1)
    
        all_mask_poe_list = ref_mask_poe_list + query_mask_poe_list
        all_mask_recon_list = ref_mask_recon_list + query_mask_recon_list
        all_data_list = ref_data_list + query_data_list
        all_mask_batch = torch.cat([ref_mask_batch, query_mask_batch],0)
        
        all_data = torch.cat([ref_data, query_data], 0)
        all_batch = torch.cat([ref_b, query_b], 0)
        all_mask_recon =torch.cat([ref_mask_recon, query_mask_recon], 0)
        all_mask_poe =torch.cat([ref_mask_poe, query_mask_poe], 0)
        all_cty =torch.cat([ref_cty, query_cty], 0)
        all_b =torch.cat([ref_b, query_b], 0)
        all_c = [torch.cat([r.reshape(-1),
                            q.to(device=r.device, dtype=r.dtype).reshape(-1)], dim=0)
                 for r, q in zip(ref_c, query_c)]
        all_train_query_info = torch.cat([
            torch.ones(ref_data.shape[0], dtype=torch.long, device=device),  # ref -> 1
            torch.zeros(query_data.shape[0], dtype=torch.long, device=device) # query -> 0
        ], dim=0)
        all_transformed_dataset = MyDataset_mask_train_query(all_data, all_mask_recon, all_mask_poe, all_cty, all_b, all_c, all_train_query_info)
        n_unique_batch = n_unique_ref_batch + n_unique_query_batch
    
    else:
        all_data = ref_data
        all_mask_recon = ref_mask_recon
        all_mask_poe = ref_mask_poe
        all_cty = ref_cty
        all_b = ref_b
        all_c = ref_c
        all_transformed_dataset = MyDataset_mask(all_data, all_mask_recon, all_mask_poe, all_cty, all_b, all_c)
        query_data = None
        query_b = None
        query_mask_poe = None
        query_metadata = None
        n_unique_batch = n_unique_ref_batch
    
    total_len = len(all_transformed_dataset)
    val_len = int(val_percentage * total_len)
    all_len = total_len - val_len
    
    all_dataset, val_dataset = random_split(all_transformed_dataset, [all_len, val_len])
    all_dl = DataLoader(all_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    generator = M3_model(n_features,
                                     batch_classify_dim=n_unique_batch, 
                                     condition_dim=n_unique_conditions, z_dim=embedding_dim, hidden_features=[embedding_dim, embedding_dim]).to(device)
    
    generator = train_M3(all_dl, val_dl, generator, criterion_smooth_cty, criterion, criterion_kl, device,
                                                       lr=lr, num_epochs=num_epochs, batch_classify_dim=n_unique_batch, condition_dim=n_unique_conditions,   min_delta = min_delta,
                                                       early_stop_patience = early_stop_patience, weight_batch_ae=weight_batch_ae, weight_modality=weight_modality, nfeatures=n_features)

    if len(preds)==2:
        return ref_data, ref_b, ref_mask_poe, ref_metadata, query_data, query_b, query_mask_poe, query_metadata, generator, preds[0], preds[1] 
    if len(preds)==1:
        return ref_data, ref_b, ref_mask_poe, ref_metadata, query_data, query_b, query_mask_poe, query_metadata, generator, preds[0], 0
    


def run_M3_update_after_imputation(modality1_path, modality2_path, modality3_path, metadata_path, save_path, condition_name, cty_name, batch_size, lr, num_epochs, 
            min_delta, early_stop_patience, val_percentage, hvg_num, weight_modality, weight_batch_ae, embedding_dim, 
            select_train_batch, select_test_batch, balance_training=False):

    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.FloatTensor 
    LongTensor = torch.LongTensor 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label, batch = load_and_merge_metadata(metadata_path)
    cty = convert_to_longtensor(label[cty_name])
    n_unique_cty = len(torch.unique(cty))
    condition = [convert_to_longtensor(label[c]) for c in condition_name]
    n_unique_conditions = [len(torch.unique(convert_to_longtensor(label[c]))) for c in condition_name]
    
    count_rna_list = load_if_available(modality1_path)
    count_adt_list = load_if_available(modality2_path)
    count_atac_list = load_if_available(modality3_path)
    count_rna, count_adt, count_atac = fill_and_concat_available_lists(count_rna_list, count_adt_list, count_atac_list)
    
    count_rna = process_count_matrix(count_rna, count_rna_list, hvg_num[0])
    count_adt = process_count_matrix(count_adt, count_adt_list, hvg_num[1])
    count_atac = process_count_matrix(count_atac, count_atac_list, hvg_num[2])

    if balance_training == True:
        ref_b_full, ref_c_full, ref_cty_full, ref_count_rna_full, ref_count_adt_full, ref_count_atac_full, ref_metadata_full = get_ref_query_data(batch, condition, cty, count_rna, count_adt, count_atac, label, select_batch=select_train_batch)
        ref_b, ref_c, ref_cty, ref_count_rna, ref_count_adt, ref_count_atac, ref_metadata = subsample_by_batch(ref_b_full, ref_c_full, ref_cty_full, ref_count_rna_full, ref_count_adt_full, ref_count_atac_full, ref_metadata_full)
    else:
        ref_b, ref_c, ref_cty, ref_count_rna, ref_count_adt, ref_count_atac, ref_metadata = get_ref_query_data(batch, condition, cty, count_rna, count_adt, count_atac, label, select_batch=select_train_batch)
        
    ref_mask_poe_list = []
    ref_mask_recon_list = []
    ref_data_list = []
    if ref_count_rna is not None:
        ref_mask_poe_list, ref_mask_recon_list, ref_data_list, ref_mask_batch = process_ref_count_after_imputation(ref_count_rna, device, ref_mask_poe_list, ref_mask_recon_list, ref_data_list)
    if ref_count_adt is not None:
        ref_mask_poe_list, ref_mask_recon_list, ref_data_list, ref_mask_batch = process_ref_count_after_imputation(ref_count_adt, device, ref_mask_poe_list, ref_mask_recon_list, ref_data_list)
    if ref_count_atac is not None:
        ref_mask_poe_list, ref_mask_recon_list, ref_data_list, ref_mask_batch = process_ref_count_after_imputation(ref_count_atac, device, ref_mask_poe_list, ref_mask_recon_list, ref_data_list)
    
    # Final concatenation
    ref_data_list = [torch.nan_to_num(d, nan=0.0) for d in ref_data_list]
    ref_mask_poe = torch.cat(ref_mask_poe_list + [ref_mask_batch[:, 1].unsqueeze(1).to(device)], dim=1)
    ref_mask_recon = torch.cat(ref_mask_recon_list, dim=1)
    ref_data = torch.cat(ref_data_list, dim=1)
    transformed_dataset = MyDataset_mask(ref_data,  ref_mask_recon, ref_mask_poe, ref_cty, ref_b, ref_c)
    
    n_unique_ref_batch = len(torch.unique(ref_b))
    n_features = [d.shape[1] for d in ref_data_list] 
    classify_dim = torch.max(cty)+1
    
    if select_test_batch!=None:
        query_b, query_c, query_cty, query_count_rna, query_count_adt, query_count_atac, query_metadata = get_ref_query_data(batch, condition, cty, count_rna, count_adt, count_atac, label, select_batch=select_test_batch)
        query_b = query_b + max(ref_b) + 1
        query_mask_poe_list = []
        query_mask_recon_list = []
        query_data_list = []
        if query_count_rna is not None:
            query_mask_poe_list, query_mask_recon_list, query_data_list, query_mask_batch = process_ref_count_after_imputation(query_count_rna, device, query_mask_poe_list, query_mask_recon_list, query_data_list)
        if query_count_adt is not None:
            query_mask_poe_list, query_mask_recon_list, query_data_list, query_mask_batch = process_ref_count_after_imputation(query_count_adt, device, query_mask_poe_list, query_mask_recon_list, query_data_list)
        if query_count_atac is not None:
            query_mask_poe_list, query_mask_recon_list, query_data_list, query_mask_batch = process_ref_count_after_imputation(query_count_atac, device, query_mask_poe_list, query_mask_recon_list, query_data_list)
    
        n_unique_query_batch = len(torch.unique(query_b))
    
        # Final concatenation
        query_data_list = [torch.nan_to_num(d, nan=0.0) for d in query_data_list]
        query_mask_poe = torch.cat(query_mask_poe_list + [query_mask_batch[:, 1].unsqueeze(1).to(device)], dim=1)
        query_mask_recon = torch.cat(query_mask_recon_list, dim=1)
        query_data = torch.cat(query_data_list, dim=1)
    
        all_mask_poe_list = ref_mask_poe_list + query_mask_poe_list
        all_mask_recon_list = ref_mask_recon_list + query_mask_recon_list
        all_data_list = ref_data_list + query_data_list
        all_mask_batch = torch.cat([ref_mask_batch, query_mask_batch],0)
        
        all_data = torch.cat([ref_data, query_data], 0)
        all_batch = torch.cat([ref_b, query_b], 0)
        all_mask_recon =torch.cat([ref_mask_recon, query_mask_recon], 0)
        all_mask_poe =torch.cat([ref_mask_poe, query_mask_poe], 0)
        all_cty =torch.cat([ref_cty, query_cty], 0)
        all_b =torch.cat([ref_b, query_b], 0)
        all_c = [torch.cat([r.reshape(-1),
                            q.to(device=r.device, dtype=r.dtype).reshape(-1)], dim=0)
                 for r, q in zip(ref_c, query_c)]
        all_train_query_info = torch.cat([
            torch.ones(ref_data.shape[0], dtype=torch.long, device=device),  # ref -> 1
            torch.zeros(query_data.shape[0], dtype=torch.long, device=device) # query -> 0
        ], dim=0)
        all_transformed_dataset = MyDataset_mask_train_query(all_data, all_mask_recon, all_mask_poe, all_cty, all_b, all_c, all_train_query_info)
        n_unique_batch = n_unique_ref_batch + n_unique_query_batch
    
    else:
        all_data = ref_data
        all_mask_recon = ref_mask_recon
        all_mask_poe = ref_mask_poe
        all_cty = ref_cty
        all_b = ref_b
        all_c = ref_c
        all_transformed_dataset = MyDataset_mask(all_data, all_mask_recon, all_mask_poe, all_cty, all_b, all_c)
        n_unique_batch = n_unique_ref_batch
    
    total_len = len(all_transformed_dataset)
    val_len = int(val_percentage * total_len)
    all_len = total_len - val_len
    
    all_dataset, val_dataset = random_split(all_transformed_dataset, [all_len, val_len])
    all_dl = DataLoader(all_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    criterion_smooth_cty = nn.CrossEntropyLoss()
    criterion = nn.MSELoss().to(device)
    criterion_kl = KL_loss()
    
    preds = [t.clone() for t in ref_c]
    for i in range(len(ref_c)):
        new_data = ref_data[:,0:1000]
        y_true = ref_c[i]
        num_classes = int(y_true.max().item() + 1)
        out = ada_self_training(
            X_tensor=new_data, y0_tensor=y_true, num_classes=num_classes,
            n_iters=5, seed_ratio=0.2, sample_ratio=0.2,
            alpha_start=1.6, alpha_end=1.0, tau=0.3, relabeled_weight=0.5,
            use_pca=True, pca_var=0.70, max_pcs=20, min_pcs=10,
            epochs_per_iter=30, batch_size=256, lr=1e-3, random_state=42,
            print_progress=True)
        preds[i] = out["final_pred"]
    
    ref_c = torch.as_tensor(preds, device=device)

    
    preds = [t.clone() for t in ref_c]
    for i in range(len(ref_c)):
        new_data = ref_data[:,0:1000]
        y_true = ref_c[i]
        num_classes = int(y_true.max().item() + 1)
        out = ada_self_training(
            X_tensor=new_data, y0_tensor=y_true, num_classes=num_classes,
            n_iters=5, seed_ratio=0.2, sample_ratio=0.2,
            alpha_start=1.6, alpha_end=1.0, tau=0.3, relabeled_weight=0.5,
            use_pca=True, pca_var=0.70, max_pcs=20, min_pcs=10,
            epochs_per_iter=30, batch_size=256, lr=1e-3, random_state=42,
            print_progress=True)
        preds[i] = out["final_pred"]
    
    ref_c = torch.as_tensor(preds, device=device)
    
    if select_test_batch!=None:
        query_b, query_c, query_cty, query_count_rna, query_count_adt, query_count_atac, query_metadata = get_ref_query_data(batch, condition, cty, count_rna, count_adt, count_atac, label, select_batch=select_test_batch)
        query_b = query_b + max(ref_b) + 1
        query_mask_poe_list = []
        query_mask_recon_list = []
        query_data_list = []
        if query_count_rna is not None:
            query_mask_poe_list, query_mask_recon_list, query_data_list, query_mask_batch = process_ref_count_after_imputation(query_count_rna, device, query_mask_poe_list, query_mask_recon_list, query_data_list)
        if query_count_adt is not None:
            query_mask_poe_list, query_mask_recon_list, query_data_list, query_mask_batch = process_ref_count_after_imputation(query_count_adt, device, query_mask_poe_list, query_mask_recon_list, query_data_list)
        if query_count_atac is not None:
            query_mask_poe_list, query_mask_recon_list, query_data_list, query_mask_batch = process_ref_count_after_imputation(query_count_atac, device, query_mask_poe_list, query_mask_recon_list, query_data_list)
    
        n_unique_query_batch = len(torch.unique(query_b))
    
        # Final concatenation
        query_data_list = [torch.nan_to_num(d, nan=0.0) for d in query_data_list]
        query_mask_poe = torch.cat(query_mask_poe_list + [query_mask_batch[:, 1].unsqueeze(1).to(device)], dim=1)
        query_mask_recon = torch.cat(query_mask_recon_list, dim=1)
        query_data = torch.cat(query_data_list, dim=1)
    
        all_mask_poe_list = ref_mask_poe_list + query_mask_poe_list
        all_mask_recon_list = ref_mask_recon_list + query_mask_recon_list
        all_data_list = ref_data_list + query_data_list
        all_mask_batch = torch.cat([ref_mask_batch, query_mask_batch],0)
        
        all_data = torch.cat([ref_data, query_data], 0)
        all_batch = torch.cat([ref_b, query_b], 0)
        all_mask_recon =torch.cat([ref_mask_recon, query_mask_recon], 0)
        all_mask_poe =torch.cat([ref_mask_poe, query_mask_poe], 0)
        all_cty =torch.cat([ref_cty, query_cty], 0)
        all_b =torch.cat([ref_b, query_b], 0)
        all_c = [torch.cat([r.reshape(-1),
                            q.to(device=r.device, dtype=r.dtype).reshape(-1)], dim=0)
                 for r, q in zip(ref_c, query_c)]
        all_train_query_info = torch.cat([
            torch.ones(ref_data.shape[0], dtype=torch.long, device=device),  # ref -> 1
            torch.zeros(query_data.shape[0], dtype=torch.long, device=device) # query -> 0
        ], dim=0)
        all_transformed_dataset = MyDataset_mask_train_query(all_data, all_mask_recon, all_mask_poe, all_cty, all_b, all_c, all_train_query_info)
        n_unique_batch = n_unique_ref_batch + n_unique_query_batch
    
    else:
        all_data = ref_data
        all_mask_recon = ref_mask_recon
        all_mask_poe = ref_mask_poe
        all_cty = ref_cty
        all_b = ref_b
        all_c = ref_c
        all_transformed_dataset = MyDataset_mask(all_data, all_mask_recon, all_mask_poe, all_cty, all_b, all_c)
        query_data = None
        query_b = None
        query_mask_poe = None
        query_metadata = None
        n_unique_batch = n_unique_ref_batch
    
    total_len = len(all_transformed_dataset)
    val_len = int(val_percentage * total_len)
    all_len = total_len - val_len
    
    all_dataset, val_dataset = random_split(all_transformed_dataset, [all_len, val_len])
    all_dl = DataLoader(all_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    generator = M3_model(n_features,
                                     batch_classify_dim=n_unique_batch, 
                                     condition_dim=n_unique_conditions, z_dim=embedding_dim, hidden_features=[embedding_dim, embedding_dim]).to(device)
    
    generator = train_M3(all_dl, val_dl, generator, criterion_smooth_cty, criterion, criterion_kl, device,
                                                       lr=lr, num_epochs=num_epochs, batch_classify_dim=n_unique_batch, condition_dim=n_unique_conditions,   min_delta = min_delta,
                                                       early_stop_patience = early_stop_patience, weight_batch_ae=weight_batch_ae, weight_modality=weight_modality, nfeatures=n_features)

    if len(preds)==2:
        return ref_data, ref_b, ref_mask_poe, ref_metadata, query_data, query_b, query_mask_poe, query_metadata, generator, preds[0], preds[1] 
    if len(preds)==1:
        return ref_data, ref_b, ref_mask_poe, ref_metadata, query_data, query_b, query_mask_poe, query_metadata, generator, preds[0], 0
    

# def run_M3_update_no_classify(modality1_path, modality2_path, modality3_path, metadata_path, save_path, condition_name, cty_name, batch_size, lr, num_epochs,
#             min_delta, early_stop_patience, val_percentage, hvg_num, weight_modality, weight_batch_ae, embedding_dim,
#             select_train_batch, select_test_batch, balance_training=False):

#     cuda = True if torch.cuda.is_available() else False
#     FloatTensor = torch.FloatTensor
#     LongTensor = torch.LongTensor
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     label, batch = load_and_merge_metadata(metadata_path)
#     cty = convert_to_longtensor(label[cty_name])
#     n_unique_cty = len(torch.unique(cty))
#     condition = [convert_to_longtensor(label[c]) for c in condition_name]
#     n_unique_conditions = [len(torch.unique(convert_to_longtensor(label[c]))) for c in condition_name]
    
#     count_rna_list = load_if_available(modality1_path)
#     count_adt_list = load_if_available(modality2_path)
#     count_atac_list = load_if_available(modality3_path)
#     count_rna, count_adt, count_atac = fill_and_concat_available_lists(count_rna_list, count_adt_list, count_atac_list)
    
#     count_rna = process_count_matrix(count_rna, count_rna_list, hvg_num[0])
#     count_adt = process_count_matrix(count_adt, count_adt_list, hvg_num[1])
#     count_atac = process_count_matrix(count_atac, count_atac_list, hvg_num[2])

#     if balance_training == True:
#         ref_b_full, ref_c_full, ref_cty_full, ref_count_rna_full, ref_count_adt_full, ref_count_atac_full, ref_metadata_full = get_ref_query_data(batch, condition, cty, count_rna, count_adt, count_atac, label, select_batch=select_train_batch)
#         ref_b, ref_c, ref_cty, ref_count_rna, ref_count_adt, ref_count_atac, ref_metadata = subsample_by_batch(ref_b_full, ref_c_full, ref_cty_full, ref_count_rna_full, ref_count_adt_full, ref_count_atac_full, ref_metadata_full)
#     else:
#         ref_b, ref_c, ref_cty, ref_count_rna, ref_count_adt, ref_count_atac, ref_metadata = get_ref_query_data(batch, condition, cty, count_rna, count_adt, count_atac, label, select_batch=select_train_batch)
        
#     ref_mask_poe_list = []
#     ref_mask_recon_list = []
#     ref_data_list = []
#     if ref_count_rna is not None:
#         ref_mask_poe_list, ref_mask_recon_list, ref_data_list, ref_mask_batch = process_ref_count(ref_count_rna, device, ref_mask_poe_list, ref_mask_recon_list, ref_data_list)
#     if ref_count_adt is not None:
#         ref_mask_poe_list, ref_mask_recon_list, ref_data_list, ref_mask_batch = process_ref_count(ref_count_adt, device, ref_mask_poe_list, ref_mask_recon_list, ref_data_list)
#     if ref_count_atac is not None:
#         ref_mask_poe_list, ref_mask_recon_list, ref_data_list, ref_mask_batch = process_ref_count(ref_count_atac, device, ref_mask_poe_list, ref_mask_recon_list, ref_data_list)
    
#     # Final concatenation
#     ref_data_list = [torch.nan_to_num(d, nan=0.0) for d in ref_data_list]
#     ref_mask_poe = torch.cat(ref_mask_poe_list + [ref_mask_batch[:, 1].unsqueeze(1).to(device)], dim=1)
#     ref_mask_recon = torch.cat(ref_mask_recon_list, dim=1)
#     ref_data = torch.cat(ref_data_list, dim=1)
#     transformed_dataset = MyDataset_mask(ref_data,  ref_mask_recon, ref_mask_poe, ref_cty, ref_b, ref_c)
    
#     n_unique_ref_batch = len(torch.unique(ref_b))
#     n_features = [d.shape[1] for d in ref_data_list]
#     classify_dim = torch.max(cty)+1
    
#     if select_test_batch!=None:
#         query_b, query_c, query_cty, query_count_rna, query_count_adt, query_count_atac, query_metadata = get_ref_query_data(batch, condition, cty, count_rna, count_adt, count_atac, label, select_batch=select_test_batch)
#         query_b = query_b + max(ref_b) + 1
#         query_mask_poe_list = []
#         query_mask_recon_list = []
#         query_data_list = []
#         if query_count_rna is not None:
#             query_mask_poe_list, query_mask_recon_list, query_data_list, query_mask_batch = process_ref_count(query_count_rna, device, query_mask_poe_list, query_mask_recon_list, query_data_list)
#         if query_count_adt is not None:
#             query_mask_poe_list, query_mask_recon_list, query_data_list, query_mask_batch = process_ref_count(query_count_adt, device, query_mask_poe_list, query_mask_recon_list, query_data_list)
#         if query_count_atac is not None:
#             query_mask_poe_list, query_mask_recon_list, query_data_list, query_mask_batch = process_ref_count(query_count_atac, device, query_mask_poe_list, query_mask_recon_list, query_data_list)
    
#         n_unique_query_batch = len(torch.unique(query_b))
    
#         # Final concatenation
#         query_data_list = [torch.nan_to_num(d, nan=0.0) for d in query_data_list]
#         query_mask_poe = torch.cat(query_mask_poe_list + [query_mask_batch[:, 1].unsqueeze(1).to(device)], dim=1)
#         query_mask_recon = torch.cat(query_mask_recon_list, dim=1)
#         query_data = torch.cat(query_data_list, dim=1)
    
#         all_mask_poe_list = ref_mask_poe_list + query_mask_poe_list
#         all_mask_recon_list = ref_mask_recon_list + query_mask_recon_list
#         all_data_list = ref_data_list + query_data_list
#         all_mask_batch = torch.cat([ref_mask_batch, query_mask_batch],0)
        
#         all_data = torch.cat([ref_data, query_data], 0)
#         all_batch = torch.cat([ref_b, query_b], 0)
#         all_mask_recon =torch.cat([ref_mask_recon, query_mask_recon], 0)
#         all_mask_poe =torch.cat([ref_mask_poe, query_mask_poe], 0)
#         all_cty =torch.cat([ref_cty, query_cty], 0)
#         all_b =torch.cat([ref_b, query_b], 0)
#         all_c = [torch.cat([r.reshape(-1),
#                             q.to(device=r.device, dtype=r.dtype).reshape(-1)], dim=0)
#                  for r, q in zip(ref_c, query_c)]
#         all_train_query_info = torch.cat([
#             torch.ones(ref_data.shape[0], dtype=torch.long, device=device),  # ref -> 1
#             torch.zeros(query_data.shape[0], dtype=torch.long, device=device) # query -> 0
#         ], dim=0)
#         all_transformed_dataset = MyDataset_mask_train_query(all_data, all_mask_recon, all_mask_poe, all_cty, all_b, all_c, all_train_query_info)
#         n_unique_batch = n_unique_ref_batch + n_unique_query_batch
    
#     else:
#         all_data = ref_data
#         all_mask_recon = ref_mask_recon
#         all_mask_poe = ref_mask_poe
#         all_cty = ref_cty
#         all_b = ref_b
#         all_c = ref_c
#         all_transformed_dataset = MyDataset_mask(all_data, all_mask_recon, all_mask_poe, all_cty, all_b, all_c)
#         n_unique_batch = n_unique_ref_batch
    
#     total_len = len(all_transformed_dataset)
#     val_len = int(val_percentage * total_len)
#     all_len = total_len - val_len
    
#     all_dataset, val_dataset = random_split(all_transformed_dataset, [all_len, val_len])
#     all_dl = DataLoader(all_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
#     val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
#     criterion_smooth_cty = nn.CrossEntropyLoss()
#     criterion = nn.MSELoss().to(device)
#     criterion_kl = KL_loss()
    
#     preds = [t.clone() for t in ref_c]
#     preds[0] = 0
#     #preds[1] = 0

    
#     if select_test_batch!=None:
#         query_b, query_c, query_cty, query_count_rna, query_count_adt, query_count_atac, query_metadata = get_ref_query_data(batch, condition, cty, count_rna, count_adt, count_atac, label, select_batch=select_test_batch)
#         query_b = query_b + max(ref_b) + 1
#         query_mask_poe_list = []
#         query_mask_recon_list = []
#         query_data_list = []
#         if query_count_rna is not None:
#             query_mask_poe_list, query_mask_recon_list, query_data_list, query_mask_batch = process_ref_count(query_count_rna, device, query_mask_poe_list, query_mask_recon_list, query_data_list)
#         if query_count_adt is not None:
#             query_mask_poe_list, query_mask_recon_list, query_data_list, query_mask_batch = process_ref_count(query_count_adt, device, query_mask_poe_list, query_mask_recon_list, query_data_list)
#         if query_count_atac is not None:
#             query_mask_poe_list, query_mask_recon_list, query_data_list, query_mask_batch = process_ref_count(query_count_atac, device, query_mask_poe_list, query_mask_recon_list, query_data_list)
    
#         n_unique_query_batch = len(torch.unique(query_b))
    
#         # Final concatenation
#         query_data_list = [torch.nan_to_num(d, nan=0.0) for d in query_data_list]
#         query_mask_poe = torch.cat(query_mask_poe_list + [query_mask_batch[:, 1].unsqueeze(1).to(device)], dim=1)
#         query_mask_recon = torch.cat(query_mask_recon_list, dim=1)
#         query_data = torch.cat(query_data_list, dim=1)
    
#         all_mask_poe_list = ref_mask_poe_list + query_mask_poe_list
#         all_mask_recon_list = ref_mask_recon_list + query_mask_recon_list
#         all_data_list = ref_data_list + query_data_list
#         all_mask_batch = torch.cat([ref_mask_batch, query_mask_batch],0)
        
#         all_data = torch.cat([ref_data, query_data], 0)
#         all_batch = torch.cat([ref_b, query_b], 0)
#         all_mask_recon =torch.cat([ref_mask_recon, query_mask_recon], 0)
#         all_mask_poe =torch.cat([ref_mask_poe, query_mask_poe], 0)
#         all_cty =torch.cat([ref_cty, query_cty], 0)
#         all_b =torch.cat([ref_b, query_b], 0)
#         all_c = [torch.cat([r.reshape(-1),
#                             q.to(device=r.device, dtype=r.dtype).reshape(-1)], dim=0)
#                  for r, q in zip(ref_c, query_c)]
#         all_train_query_info = torch.cat([
#             torch.ones(ref_data.shape[0], dtype=torch.long, device=device),  # ref -> 1
#             torch.zeros(query_data.shape[0], dtype=torch.long, device=device) # query -> 0
#         ], dim=0)
#         all_transformed_dataset = MyDataset_mask_train_query(all_data, all_mask_recon, all_mask_poe, all_cty, all_b, all_c, all_train_query_info)
#         n_unique_batch = n_unique_ref_batch + n_unique_query_batch
    
#     else:
#         all_data = ref_data
#         all_mask_recon = ref_mask_recon
#         all_mask_poe = ref_mask_poe
#         all_cty = ref_cty
#         all_b = ref_b
#         all_c = ref_c
#         all_transformed_dataset = MyDataset_mask(all_data, all_mask_recon, all_mask_poe, all_cty, all_b, all_c)
#         query_data = None
#         query_b = None
#         query_mask_poe = None
#         query_metadata = None
#         n_unique_batch = n_unique_ref_batch
    
#     total_len = len(all_transformed_dataset)
#     val_len = int(val_percentage * total_len)
#     all_len = total_len - val_len
    
#     all_dataset, val_dataset = random_split(all_transformed_dataset, [all_len, val_len])
#     all_dl = DataLoader(all_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
#     val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
#     generator = M3_model(n_features,
#                                      batch_classify_dim=n_unique_batch,
#                                      condition_dim=n_unique_conditions, z_dim=embedding_dim, hidden_features=[embedding_dim, embedding_dim]).to(device)
    
#     generator = train_M3(all_dl, val_dl, generator, criterion_smooth_cty, criterion, criterion_kl, device,
#                                                        lr=lr, num_epochs=num_epochs, batch_classify_dim=n_unique_batch, condition_dim=n_unique_conditions,   min_delta = min_delta,
#                                                        early_stop_patience = early_stop_patience, weight_batch_ae=weight_batch_ae, weight_modality=weight_modality, nfeatures=n_features)

#     if len(preds)==2:
#         return ref_data, ref_b, ref_mask_poe, ref_metadata, query_data, query_b, query_mask_poe, query_metadata, generator, preds[0], preds[1]
#     if len(preds)==1:
#         return ref_data, ref_b, ref_mask_poe, ref_metadata, query_data, query_b, query_mask_poe, query_metadata, generator, preds[0], 0
    


def run_M3_update_no_classify(modality1_path, modality2_path, modality3_path, metadata_path, save_path, condition_name, cty_name, batch_size, lr, num_epochs,
            min_delta, early_stop_patience, val_percentage, hvg_num, weight_modality, weight_batch_ae, embedding_dim,
            select_train_batch, select_test_batch, balance_training=False):

    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label, batch = load_and_merge_metadata(metadata_path)
    cty = convert_to_longtensor(label[cty_name])
    n_unique_cty = len(torch.unique(cty))
    condition = [convert_to_longtensor(label[c]) for c in condition_name]
    n_unique_conditions = [len(torch.unique(convert_to_longtensor(label[c]))) for c in condition_name]

    count_rna_list = load_if_available(modality1_path)
    count_adt_list = load_if_available(modality2_path)
    count_atac_list = load_if_available(modality3_path)
    count_rna, count_adt, count_atac = fill_and_concat_available_lists(count_rna_list, count_adt_list, count_atac_list)

    count_rna = process_count_matrix(count_rna, count_rna_list, hvg_num[0])
    count_adt = process_count_matrix(count_adt, count_adt_list, hvg_num[1])
    count_atac = process_count_matrix(count_atac, count_atac_list, hvg_num[2])

    if balance_training == True:
        ref_b_full, ref_c_full, ref_cty_full, ref_count_rna_full, ref_count_adt_full, ref_count_atac_full, ref_metadata_full = get_ref_query_data(batch, condition, cty, count_rna, count_adt, count_atac, label, select_batch=select_train_batch)
        ref_b, ref_c, ref_cty, ref_count_rna, ref_count_adt, ref_count_atac, ref_metadata = subsample_by_batch(ref_b_full, ref_c_full, ref_cty_full, ref_count_rna_full, ref_count_adt_full, ref_count_atac_full, ref_metadata_full)
    else:
        ref_b, ref_c, ref_cty, ref_count_rna, ref_count_adt, ref_count_atac, ref_metadata = get_ref_query_data(batch, condition, cty, count_rna, count_adt, count_atac, label, select_batch=select_train_batch)

    ref_mask_poe_list = []
    ref_mask_recon_list = []
    ref_data_list = []
    if ref_count_rna is not None:
        ref_mask_poe_list, ref_mask_recon_list, ref_data_list, ref_mask_batch = process_ref_count(ref_count_rna, device, ref_mask_poe_list, ref_mask_recon_list, ref_data_list)
    if ref_count_adt is not None:
        ref_mask_poe_list, ref_mask_recon_list, ref_data_list, ref_mask_batch = process_ref_count(ref_count_adt, device, ref_mask_poe_list, ref_mask_recon_list, ref_data_list)
    if ref_count_atac is not None:
        ref_mask_poe_list, ref_mask_recon_list, ref_data_list, ref_mask_batch = process_ref_count(ref_count_atac, device, ref_mask_poe_list, ref_mask_recon_list, ref_data_list)

    # Final concatenation
    ref_data_list = [torch.nan_to_num(d, nan=0.0) for d in ref_data_list]
    ref_mask_poe = torch.cat(ref_mask_poe_list + [ref_mask_batch[:, 1].unsqueeze(1).to(device)], dim=1)
    ref_mask_recon = torch.cat(ref_mask_recon_list, dim=1)
    ref_data = torch.cat(ref_data_list, dim=1)
    transformed_dataset = MyDataset_mask(ref_data,  ref_mask_recon, ref_mask_poe, ref_cty, ref_b, ref_c)

    n_unique_ref_batch = len(torch.unique(ref_b))
    n_features = [d.shape[1] for d in ref_data_list]
    classify_dim = torch.max(cty)+1

    criterion_smooth_cty = nn.CrossEntropyLoss()
    criterion = nn.MSELoss().to(device)
    criterion_kl = KL_loss()

    preds = [t.clone() for t in ref_c]
    preds[0] = 0
    #preds[1] = 0

    if select_test_batch!=None:
        query_b, query_c, query_cty, query_count_rna, query_count_adt, query_count_atac, query_metadata = get_ref_query_data(batch, condition, cty, count_rna, count_adt, count_atac, label, select_batch=select_test_batch)
        query_b = query_b + max(ref_b) + 1
        query_mask_poe_list = []
        query_mask_recon_list = []
        query_data_list = []
        if query_count_rna is not None:
            query_mask_poe_list, query_mask_recon_list, query_data_list, query_mask_batch = process_ref_count(query_count_rna, device, query_mask_poe_list, query_mask_recon_list, query_data_list)
        if query_count_adt is not None:
            query_mask_poe_list, query_mask_recon_list, query_data_list, query_mask_batch = process_ref_count(query_count_adt, device, query_mask_poe_list, query_mask_recon_list, query_data_list)
        if query_count_atac is not None:
            query_mask_poe_list, query_mask_recon_list, query_data_list, query_mask_batch = process_ref_count(query_count_atac, device, query_mask_poe_list, query_mask_recon_list, query_data_list)

        n_unique_query_batch = len(torch.unique(query_b))

        # Final concatenation
        query_data_list = [torch.nan_to_num(d, nan=0.0) for d in query_data_list]
        query_mask_poe = torch.cat(query_mask_poe_list + [query_mask_batch[:, 1].unsqueeze(1).to(device)], dim=1)
        query_mask_recon = torch.cat(query_mask_recon_list, dim=1)
        query_data = torch.cat(query_data_list, dim=1)

        all_mask_poe_list = ref_mask_poe_list + query_mask_poe_list
        all_mask_recon_list = ref_mask_recon_list + query_mask_recon_list
        all_data_list = ref_data_list + query_data_list
        all_mask_batch = torch.cat([ref_mask_batch, query_mask_batch],0)

        all_data = torch.cat([ref_data, query_data], 0)
        all_batch = torch.cat([ref_b, query_b], 0)
        all_mask_recon =torch.cat([ref_mask_recon, query_mask_recon], 0)
        all_mask_poe =torch.cat([ref_mask_poe, query_mask_poe], 0)
        all_cty =torch.cat([ref_cty, query_cty], 0)
        all_b =torch.cat([ref_b, query_b], 0)
        all_c = [torch.cat([r.reshape(-1),
                            q.to(device=r.device, dtype=r.dtype).reshape(-1)], dim=0)
                 for r, q in zip(ref_c, query_c)]
        all_train_query_info = torch.cat([
            torch.ones(ref_data.shape[0], dtype=torch.long, device=device),  # ref -> 1
            torch.zeros(query_data.shape[0], dtype=torch.long, device=device) # query -> 0
        ], dim=0)
        all_transformed_dataset = MyDataset_mask_train_query(all_data, all_mask_recon, all_mask_poe, all_cty, all_b, all_c, all_train_query_info)
        n_unique_batch = n_unique_ref_batch + n_unique_query_batch

    else:
        all_data = ref_data
        all_mask_recon = ref_mask_recon
        all_mask_poe = ref_mask_poe
        all_cty = ref_cty
        all_b = ref_b
        all_c = ref_c
        all_transformed_dataset = MyDataset_mask(all_data, all_mask_recon, all_mask_poe, all_cty, all_b, all_c)
        query_data = None
        query_b = None
        query_mask_poe = None
        query_metadata = None
        n_unique_batch = n_unique_ref_batch

    total_len = len(all_transformed_dataset)
    val_len = int(val_percentage * total_len)
    all_len = total_len - val_len

    all_dataset, val_dataset = random_split(all_transformed_dataset, [all_len, val_len])
    all_dl = DataLoader(all_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    generator = M3_model(n_features,
                                     batch_classify_dim=n_unique_batch,
                                     condition_dim=n_unique_conditions, z_dim=embedding_dim, hidden_features=[embedding_dim, embedding_dim]).to(device)

    generator = train_M3(all_dl, val_dl, generator, criterion_smooth_cty, criterion, criterion_kl, device,
                                                       lr=lr, num_epochs=num_epochs, batch_classify_dim=n_unique_batch, condition_dim=n_unique_conditions,   min_delta = min_delta,
                                                       early_stop_patience = early_stop_patience, weight_batch_ae=weight_batch_ae, weight_modality=weight_modality, nfeatures=n_features)

    if len(preds)==2:
        return ref_data, ref_b, ref_mask_poe, ref_metadata, query_data, query_b, query_mask_poe, query_metadata, generator, preds[0], preds[1]
    if len(preds)==1:
        return ref_data, ref_b, ref_mask_poe, ref_metadata, query_data, query_b, query_mask_poe, query_metadata, generator, preds[0], 0
    



# def run_M3_update_with_query(modality1_path, modality2_path, modality3_path, metadata_path, save_path, condition_name, cty_name, batch_size, lr, num_epochs,
#             min_delta, early_stop_patience, val_percentage, hvg_num, weight_modality, weight_batch_ae, embedding_dim,
#             select_train_batch, select_test_batch, balance_training=False):

#     cuda = True if torch.cuda.is_available() else False
#     FloatTensor = torch.FloatTensor
#     LongTensor = torch.LongTensor
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     label, batch = load_and_merge_metadata(metadata_path)
#     cty = convert_to_longtensor(label[cty_name])
#     n_unique_cty = len(torch.unique(cty))
#     condition = [convert_to_longtensor(label[c]) for c in condition_name]
#     n_unique_conditions = [len(torch.unique(convert_to_longtensor(label[c]))) for c in condition_name]
    
#     count_rna_list = load_if_available(modality1_path)
#     count_adt_list = load_if_available(modality2_path)
#     count_atac_list = load_if_available(modality3_path)
#     count_rna, count_adt, count_atac = fill_and_concat_available_lists(count_rna_list, count_adt_list, count_atac_list)
    
#     count_rna = process_count_matrix(count_rna, count_rna_list, hvg_num[0])
#     count_adt = process_count_matrix(count_adt, count_adt_list, hvg_num[1])
#     count_atac = process_count_matrix(count_atac, count_atac_list, hvg_num[2])

#     if balance_training == True:
#         ref_b_full, ref_c_full, ref_cty_full, ref_count_rna_full, ref_count_adt_full, ref_count_atac_full, ref_metadata_full = get_ref_query_data(batch, condition, cty, count_rna, count_adt, count_atac, label, select_batch=select_train_batch)
#         ref_b, ref_c, ref_cty, ref_count_rna, ref_count_adt, ref_count_atac, ref_metadata = subsample_by_batch(ref_b_full, ref_c_full, ref_cty_full, ref_count_rna_full, ref_count_adt_full, ref_count_atac_full, ref_metadata_full)
#     else:
#         ref_b, ref_c, ref_cty, ref_count_rna, ref_count_adt, ref_count_atac, ref_metadata = get_ref_query_data(batch, condition, cty, count_rna, count_adt, count_atac, label, select_batch=select_train_batch)
        
#     ref_mask_poe_list = []
#     ref_mask_recon_list = []
#     ref_data_list = []
#     if ref_count_rna is not None:
#         ref_mask_poe_list, ref_mask_recon_list, ref_data_list, ref_mask_batch = process_ref_count(ref_count_rna, device, ref_mask_poe_list, ref_mask_recon_list, ref_data_list)
#     if ref_count_adt is not None:
#         ref_mask_poe_list, ref_mask_recon_list, ref_data_list, ref_mask_batch = process_ref_count(ref_count_adt, device, ref_mask_poe_list, ref_mask_recon_list, ref_data_list)
#     if ref_count_atac is not None:
#         ref_mask_poe_list, ref_mask_recon_list, ref_data_list, ref_mask_batch = process_ref_count(ref_count_atac, device, ref_mask_poe_list, ref_mask_recon_list, ref_data_list)
    
#     # Final concatenation
#     ref_data_list = [torch.nan_to_num(d, nan=0.0) for d in ref_data_list]
#     ref_mask_poe = torch.cat(ref_mask_poe_list + [ref_mask_batch[:, 1].unsqueeze(1).to(device)], dim=1)
#     ref_mask_recon = torch.cat(ref_mask_recon_list, dim=1)
#     ref_data = torch.cat(ref_data_list, dim=1)
#     transformed_dataset = MyDataset_mask(ref_data,  ref_mask_recon, ref_mask_poe, ref_cty, ref_b, ref_c)
    
#     n_unique_ref_batch = len(torch.unique(ref_b))
#     n_features = [d.shape[1] for d in ref_data_list]
#     classify_dim = torch.max(cty)+1
    
#     if select_test_batch!=None:
#         query_b, query_c, query_cty, query_count_rna, query_count_adt, query_count_atac, query_metadata = get_ref_query_data(batch, condition, cty, count_rna, count_adt, count_atac, label, select_batch=select_test_batch)
#         query_b = query_b + max(ref_b) + 1
#         query_mask_poe_list = []
#         query_mask_recon_list = []
#         query_data_list = []
#         if query_count_rna is not None:
#             query_mask_poe_list, query_mask_recon_list, query_data_list, query_mask_batch = process_ref_count(query_count_rna, device, query_mask_poe_list, query_mask_recon_list, query_data_list)
#         if query_count_adt is not None:
#             query_mask_poe_list, query_mask_recon_list, query_data_list, query_mask_batch = process_ref_count(query_count_adt, device, query_mask_poe_list, query_mask_recon_list, query_data_list)
#         if query_count_atac is not None:
#             query_mask_poe_list, query_mask_recon_list, query_data_list, query_mask_batch = process_ref_count(query_count_atac, device, query_mask_poe_list, query_mask_recon_list, query_data_list)
    
#         n_unique_query_batch = len(torch.unique(query_b))
    
#         # Final concatenation
#         query_data_list = [torch.nan_to_num(d, nan=0.0) for d in query_data_list]
#         query_mask_poe = torch.cat(query_mask_poe_list + [query_mask_batch[:, 1].unsqueeze(1).to(device)], dim=1)
#         query_mask_recon = torch.cat(query_mask_recon_list, dim=1)
#         query_data = torch.cat(query_data_list, dim=1)
    
#         all_mask_poe_list = ref_mask_poe_list + query_mask_poe_list
#         all_mask_recon_list = ref_mask_recon_list + query_mask_recon_list
#         all_data_list = ref_data_list + query_data_list
#         all_mask_batch = torch.cat([ref_mask_batch, query_mask_batch],0)
        
#         all_data = torch.cat([ref_data, query_data], 0)
#         all_batch = torch.cat([ref_b, query_b], 0)
#         all_mask_recon =torch.cat([ref_mask_recon, query_mask_recon], 0)
#         all_mask_poe =torch.cat([ref_mask_poe, query_mask_poe], 0)
#         all_cty =torch.cat([ref_cty, query_cty], 0)
#         all_b =torch.cat([ref_b, query_b], 0)
#         all_c = [torch.cat([r.reshape(-1),
#                             q.to(device=r.device, dtype=r.dtype).reshape(-1)], dim=0)
#                  for r, q in zip(ref_c, query_c)]
#         all_train_query_info = torch.cat([
#             torch.ones(ref_data.shape[0], dtype=torch.long, device=device),  # ref -> 1
#             torch.zeros(query_data.shape[0], dtype=torch.long, device=device) # query -> 0
#         ], dim=0)
#         all_transformed_dataset = MyDataset_mask_train_query(all_data, all_mask_recon, all_mask_poe, all_cty, all_b, all_c, all_train_query_info)
#         n_unique_batch = n_unique_ref_batch + n_unique_query_batch
    
#     else:
#         all_data = ref_data
#         all_mask_recon = ref_mask_recon
#         all_mask_poe = ref_mask_poe
#         all_cty = ref_cty
#         all_b = ref_b
#         all_c = ref_c
#         all_transformed_dataset = MyDataset_mask(all_data, all_mask_recon, all_mask_poe, all_cty, all_b, all_c)
#         n_unique_batch = n_unique_ref_batch
    
#     total_len = len(all_transformed_dataset)
#     val_len = int(val_percentage * total_len)
#     all_len = total_len - val_len
    
#     all_dataset, val_dataset = random_split(all_transformed_dataset, [all_len, val_len])
#     all_dl = DataLoader(all_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
#     val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
#     criterion_smooth_cty = nn.CrossEntropyLoss()
#     criterion = nn.MSELoss().to(device)
#     criterion_kl = KL_loss()
    
#     preds = [t.clone() for t in ref_c]
#     preds[0] = 0
#     #preds[1] = 0

    
#     if select_test_batch!=None:
#         query_b, query_c, query_cty, query_count_rna, query_count_adt, query_count_atac, query_metadata = get_ref_query_data(batch, condition, cty, count_rna, count_adt, count_atac, label, select_batch=select_test_batch)
#         query_b = query_b + max(ref_b) + 1
#         query_mask_poe_list = []
#         query_mask_recon_list = []
#         query_data_list = []
#         if query_count_rna is not None:
#             query_mask_poe_list, query_mask_recon_list, query_data_list, query_mask_batch = process_ref_count(query_count_rna, device, query_mask_poe_list, query_mask_recon_list, query_data_list)
#         if query_count_adt is not None:
#             query_mask_poe_list, query_mask_recon_list, query_data_list, query_mask_batch = process_ref_count(query_count_adt, device, query_mask_poe_list, query_mask_recon_list, query_data_list)
#         if query_count_atac is not None:
#             query_mask_poe_list, query_mask_recon_list, query_data_list, query_mask_batch = process_ref_count(query_count_atac, device, query_mask_poe_list, query_mask_recon_list, query_data_list)
    
#         n_unique_query_batch = len(torch.unique(query_b))
    
#         # Final concatenation
#         query_data_list = [torch.nan_to_num(d, nan=0.0) for d in query_data_list]
#         query_mask_poe = torch.cat(query_mask_poe_list + [query_mask_batch[:, 1].unsqueeze(1).to(device)], dim=1)
#         query_mask_recon = torch.cat(query_mask_recon_list, dim=1)
#         query_data = torch.cat(query_data_list, dim=1)
    
#         all_mask_poe_list = ref_mask_poe_list + query_mask_poe_list
#         all_mask_recon_list = ref_mask_recon_list + query_mask_recon_list
#         all_data_list = ref_data_list + query_data_list
#         all_mask_batch = torch.cat([ref_mask_batch, query_mask_batch],0)
        
#         all_data = torch.cat([ref_data, query_data], 0)
#         all_batch = torch.cat([ref_b, query_b], 0)
#         all_mask_recon =torch.cat([ref_mask_recon, query_mask_recon], 0)
#         all_mask_poe =torch.cat([ref_mask_poe, query_mask_poe], 0)
#         all_cty =torch.cat([ref_cty, query_cty], 0)
#         all_b =torch.cat([ref_b, query_b], 0)
#         all_c = [torch.cat([r.reshape(-1),
#                             q.to(device=r.device, dtype=r.dtype).reshape(-1)], dim=0)
#                  for r, q in zip(ref_c, query_c)]
#         all_train_query_info = torch.cat([
#             torch.ones(ref_data.shape[0], dtype=torch.long, device=device),  # ref -> 1
#             torch.zeros(query_data.shape[0], dtype=torch.long, device=device) # query -> 0
#         ], dim=0)
#         all_transformed_dataset = MyDataset_mask_train_query(all_data, all_mask_recon, all_mask_poe, all_cty, all_b, all_c, all_train_query_info)
#         n_unique_batch = n_unique_ref_batch + n_unique_query_batch
    
#     else:
#         all_data = ref_data
#         all_mask_recon = ref_mask_recon
#         all_mask_poe = ref_mask_poe
#         all_cty = ref_cty
#         all_b = ref_b
#         all_c = ref_c
#         all_transformed_dataset = MyDataset_mask(all_data, all_mask_recon, all_mask_poe, all_cty, all_b, all_c)
#         query_data = None
#         query_b = None
#         query_mask_poe = None
#         query_metadata = None
#         n_unique_batch = n_unique_ref_batch
    
#     total_len = len(all_transformed_dataset)
#     val_len = int(val_percentage * total_len)
#     all_len = total_len - val_len
    
#     all_dataset, val_dataset = random_split(all_transformed_dataset, [all_len, val_len])
#     all_dl = DataLoader(all_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
#     val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
#     generator = M3_model(n_features,
#                                      batch_classify_dim=n_unique_batch,
#                                      condition_dim=n_unique_conditions, z_dim=embedding_dim, hidden_features=[embedding_dim, embedding_dim]).to(device)
    
#     generator = train_M3_with_query(all_dl, val_dl, generator, criterion_smooth_cty, criterion, criterion_kl, device,
#                                                        lr=lr, num_epochs=num_epochs, batch_classify_dim=n_unique_batch, condition_dim=n_unique_conditions,   min_delta = min_delta,
#                                                        early_stop_patience = early_stop_patience, weight_batch_ae=weight_batch_ae, weight_modality=weight_modality, nfeatures=n_features)

#     if len(preds)==2:
#         return ref_data, ref_b, ref_mask_poe, ref_metadata, query_data, query_b, query_mask_poe, query_metadata, generator, preds[0], preds[1]
#     if len(preds)==1:
#         return ref_data, ref_b, ref_mask_poe, ref_metadata, query_data, query_b, query_mask_poe, query_metadata, generator, preds[0], 0
    

def run_M3_update_with_query(modality1_path, modality2_path, modality3_path, metadata_path, save_path, condition_name, cty_name, batch_size, lr, num_epochs,
            min_delta, early_stop_patience, val_percentage, hvg_num, weight_modality, weight_batch_ae, embedding_dim,
            select_train_batch, select_test_batch, balance_training=False,
            held_out_samples=None, donor_name=None):

    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label, batch = load_and_merge_metadata(metadata_path)
    cty = convert_to_longtensor(label[cty_name])
    n_unique_cty = len(torch.unique(cty))
    condition = [convert_to_longtensor(label[c]) for c in condition_name]
    n_unique_conditions = [len(torch.unique(convert_to_longtensor(label[c]))) for c in condition_name]

    # Donor-level held-out: a per-cell query mask (the held-out samples may span
    # several batches). When set it overrides the batch-based ref/query split;
    # the batch column is left intact so the donor adversary still removes batch.
    if held_out_samples is not None and donor_name is not None:
        _hs = set(str(s) for s in held_out_samples)
        query_keep = torch.tensor(label[donor_name].astype(str).isin(_hs).values, dtype=torch.bool)
        ref_keep = ~query_keep
    else:
        query_keep = ref_keep = None

    count_rna_list = load_if_available(modality1_path)
    count_adt_list = load_if_available(modality2_path)
    count_atac_list = load_if_available(modality3_path)
    count_rna, count_adt, count_atac = fill_and_concat_available_lists(count_rna_list, count_adt_list, count_atac_list)

    count_rna = process_count_matrix(count_rna, count_rna_list, hvg_num[0])
    count_adt = process_count_matrix(count_adt, count_adt_list, hvg_num[1])
    count_atac = process_count_matrix(count_atac, count_atac_list, hvg_num[2])

    if balance_training == True:
        ref_b_full, ref_c_full, ref_cty_full, ref_count_rna_full, ref_count_adt_full, ref_count_atac_full, ref_metadata_full = get_ref_query_data(batch, condition, cty, count_rna, count_adt, count_atac, label, select_batch=select_train_batch, keep_mask=ref_keep)
        ref_b, ref_c, ref_cty, ref_count_rna, ref_count_adt, ref_count_atac, ref_metadata = subsample_by_batch(ref_b_full, ref_c_full, ref_cty_full, ref_count_rna_full, ref_count_adt_full, ref_count_atac_full, ref_metadata_full)
    else:
        ref_b, ref_c, ref_cty, ref_count_rna, ref_count_adt, ref_count_atac, ref_metadata = get_ref_query_data(batch, condition, cty, count_rna, count_adt, count_atac, label, select_batch=select_train_batch, keep_mask=ref_keep)

    ref_mask_poe_list = []
    ref_mask_recon_list = []
    ref_data_list = []
    if ref_count_rna is not None:
        ref_mask_poe_list, ref_mask_recon_list, ref_data_list, ref_mask_batch = process_ref_count(ref_count_rna, device, ref_mask_poe_list, ref_mask_recon_list, ref_data_list)
    if ref_count_adt is not None:
        ref_mask_poe_list, ref_mask_recon_list, ref_data_list, ref_mask_batch = process_ref_count(ref_count_adt, device, ref_mask_poe_list, ref_mask_recon_list, ref_data_list)
    if ref_count_atac is not None:
        ref_mask_poe_list, ref_mask_recon_list, ref_data_list, ref_mask_batch = process_ref_count(ref_count_atac, device, ref_mask_poe_list, ref_mask_recon_list, ref_data_list)

    # Final concatenation
    ref_data_list = [torch.nan_to_num(d, nan=0.0) for d in ref_data_list]
    ref_mask_poe = torch.cat(ref_mask_poe_list + [ref_mask_batch[:, 1].unsqueeze(1).to(device)], dim=1)
    ref_mask_recon = torch.cat(ref_mask_recon_list, dim=1)
    ref_data = torch.cat(ref_data_list, dim=1)
    transformed_dataset = MyDataset_mask(ref_data,  ref_mask_recon, ref_mask_poe, ref_cty, ref_b, ref_c)

    n_unique_ref_batch = len(torch.unique(ref_b))
    n_features = [d.shape[1] for d in ref_data_list]
    classify_dim = torch.max(cty)+1

    criterion_smooth_cty = nn.CrossEntropyLoss()
    criterion = nn.MSELoss().to(device)
    criterion_kl = KL_loss()

    preds = [t.clone() for t in ref_c]
    preds[0] = 0
    #preds[1] = 0

    if select_test_batch is not None or query_keep is not None:
        query_b, query_c, query_cty, query_count_rna, query_count_adt, query_count_atac, query_metadata = get_ref_query_data(batch, condition, cty, count_rna, count_adt, count_atac, label, select_batch=select_test_batch, keep_mask=query_keep)
        query_b = query_b + max(ref_b) + 1
        query_mask_poe_list = []
        query_mask_recon_list = []
        query_data_list = []
        if query_count_rna is not None:
            query_mask_poe_list, query_mask_recon_list, query_data_list, query_mask_batch = process_ref_count(query_count_rna, device, query_mask_poe_list, query_mask_recon_list, query_data_list)
        if query_count_adt is not None:
            query_mask_poe_list, query_mask_recon_list, query_data_list, query_mask_batch = process_ref_count(query_count_adt, device, query_mask_poe_list, query_mask_recon_list, query_data_list)
        if query_count_atac is not None:
            query_mask_poe_list, query_mask_recon_list, query_data_list, query_mask_batch = process_ref_count(query_count_atac, device, query_mask_poe_list, query_mask_recon_list, query_data_list)

        n_unique_query_batch = len(torch.unique(query_b))

        # Final concatenation
        query_data_list = [torch.nan_to_num(d, nan=0.0) for d in query_data_list]
        query_mask_poe = torch.cat(query_mask_poe_list + [query_mask_batch[:, 1].unsqueeze(1).to(device)], dim=1)
        query_mask_recon = torch.cat(query_mask_recon_list, dim=1)
        query_data = torch.cat(query_data_list, dim=1)

        all_mask_poe_list = ref_mask_poe_list + query_mask_poe_list
        all_mask_recon_list = ref_mask_recon_list + query_mask_recon_list
        all_data_list = ref_data_list + query_data_list
        all_mask_batch = torch.cat([ref_mask_batch, query_mask_batch],0)

        all_data = torch.cat([ref_data, query_data], 0)
        all_batch = torch.cat([ref_b, query_b], 0)
        all_mask_recon =torch.cat([ref_mask_recon, query_mask_recon], 0)
        all_mask_poe =torch.cat([ref_mask_poe, query_mask_poe], 0)
        all_cty =torch.cat([ref_cty, query_cty], 0)
        all_b =torch.cat([ref_b, query_b], 0)
        all_c = [torch.cat([r.reshape(-1),
                            q.to(device=r.device, dtype=r.dtype).reshape(-1)], dim=0)
                 for r, q in zip(ref_c, query_c)]
        all_train_query_info = torch.cat([
            torch.ones(ref_data.shape[0], dtype=torch.long, device=device),  # ref -> 1
            torch.zeros(query_data.shape[0], dtype=torch.long, device=device) # query -> 0
        ], dim=0)
        all_transformed_dataset = MyDataset_mask_train_query(all_data, all_mask_recon, all_mask_poe, all_cty, all_b, all_c, all_train_query_info)
        n_unique_batch = n_unique_ref_batch + n_unique_query_batch

    else:
        all_data = ref_data
        all_mask_recon = ref_mask_recon
        all_mask_poe = ref_mask_poe
        all_cty = ref_cty
        all_b = ref_b
        all_c = ref_c
        all_transformed_dataset = MyDataset_mask(all_data, all_mask_recon, all_mask_poe, all_cty, all_b, all_c)
        query_data = None
        query_b = None
        query_mask_poe = None
        query_metadata = None
        n_unique_batch = n_unique_ref_batch

    total_len = len(all_transformed_dataset)
    val_len = int(val_percentage * total_len)
    all_len = total_len - val_len

    all_dataset, val_dataset = random_split(all_transformed_dataset, [all_len, val_len])
    all_dl = DataLoader(all_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    generator = M3_model(n_features,
                                     batch_classify_dim=n_unique_batch,
                                     condition_dim=n_unique_conditions, z_dim=embedding_dim, hidden_features=[embedding_dim, embedding_dim]).to(device)

    generator = train_M3_with_query(all_dl, val_dl, generator, criterion_smooth_cty, criterion, criterion_kl, device,
                                                       lr=lr, num_epochs=num_epochs, batch_classify_dim=n_unique_batch, condition_dim=n_unique_conditions,   min_delta = min_delta,
                                                       early_stop_patience = early_stop_patience, weight_batch_ae=weight_batch_ae, weight_modality=weight_modality, nfeatures=n_features)

    if len(preds)==2:
        return ref_data, ref_b, ref_mask_poe, ref_metadata, query_data, query_b, query_mask_poe, query_metadata, generator, preds[0], preds[1]
    if len(preds)==1:
        return ref_data, ref_b, ref_mask_poe, ref_metadata, query_data, query_b, query_mask_poe, query_metadata, generator, preds[0], 0