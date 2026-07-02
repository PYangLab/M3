import os
import h5py
import scipy
import random
import anndata
import numpy as np
import pandas as pd
import scanpy as sc

import torch
import torch.nn as nn
from torch import exp, log
from torch import linalg as LA
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

cuda = False 
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = False
     os.environ['PYTHONHASHSEED']=str(seed)

class MyDataset_mask(Dataset):
    def __init__(self, data, mask_recon, mask_poe, label, batch, c1, c2):
        self.data = data
        self.mask_recon = mask_recon
        self.mask_poe = mask_poe
        self.label = label
        self.batch = batch
        self.c1 = c1
        self.c2 = c2
    def __getitem__(self, index):
        img, mask_recon, mask_poe, target, batch, c1, c2 = self.data[index,:], self.mask_recon[index,:], self.mask_poe[index,:], self.label[index], self.batch[index], self.c1[index], self.c2[index]
        sample = {'data': img, 'mask_recon': mask_recon, 'mask_poe':mask_poe, 'label': target, 'batch': batch, 'c1': c1, 'c2': c2}
        return sample
    def __len__(self):
        return len(self.data)


class MyDataset_mask(Dataset):
    def __init__(self, data, mask_recon, mask_poe, label, batch, condition_list):
        self.data = data
        self.mask_recon = mask_recon
        self.mask_poe = mask_poe
        self.label = label
        self.batch = batch
        self.condition_list = condition_list  # list of tensors

    def __getitem__(self, index):
        img = self.data[index, :]
        mask_recon = self.mask_recon[index, :]
        mask_poe = self.mask_poe[index, :]
        target = self.label[index]
        batch = self.batch[index]
        conditions = [c[index] for c in self.condition_list]  # extract all condition values

        sample = {
            'data': img,
            'mask_recon': mask_recon,
            'mask_poe': mask_poe,
            'label': target,
            'batch': batch,
            'conditions': conditions  # return as a list
        }
        return sample

    def __len__(self):
        return len(self.data)


class MyDataset_mask_train_query(Dataset):
    def __init__(self, data, mask_recon, mask_poe, label, batch, condition_list, all_train_query_info):
        self.data = data
        self.mask_recon = mask_recon
        self.mask_poe = mask_poe
        self.label = label
        self.batch = batch
        self.condition_list = condition_list  # list of tensors
        self.all_train_query_info = all_train_query_info  # list of tensors

    def __getitem__(self, index):
        img = self.data[index, :]
        mask_recon = self.mask_recon[index, :]
        mask_poe = self.mask_poe[index, :]
        target = self.label[index]
        batch = self.batch[index]
        conditions = [c[index] for c in self.condition_list]  # extract all condition values
        all_train_query_info = self.all_train_query_info[index]

        sample = {
            'data': img,
            'mask_recon': mask_recon,
            'mask_poe': mask_poe,
            'label': target,
            'batch': batch,
            'conditions': conditions,  # return as a list
            'all_train_query_info': all_train_query_info
        }
        return sample

    def __len__(self):
        return len(self.data)



def read_h5_data(data_path):
    with h5py.File(data_path, "r") as f:
        if "matrix/data" in f:
            print("Using 'matrix/data'")
            h5_data = f["matrix/data"]
        elif "data" in f:
            print("Using 'data'")
            h5_data = f["data"]
        else:
            raise KeyError("Neither 'matrix/data' nor 'data' found in the H5 file")

        sparse_data = scipy.sparse.csr_matrix(np.array(h5_data).T)
        data_fs = torch.from_numpy(sparse_data.toarray()).float()
        data_fs = Variable(data_fs)
        print(data_fs.shape)
    return data_fs


def read_fs_label(label_path):
    label_fs = pd.read_csv(label_path,header=None,index_col=False)  #
    label_fs = pd.Categorical(label_fs.iloc[1:(label_fs.shape[0]),0]).codes
    label_fs = np.array(label_fs[:]).astype('int32')
    label_fs = torch.from_numpy(label_fs)#
    label_fs = label_fs.type(LongTensor)
    return label_fs

def compute_zscore(data):
    temp_mean = torch.mean(data,1)
    temp_mean = temp_mean.repeat(data.size(1),1).transpose(0,1)
    temp_std = torch.std(data,1)
    temp_std = temp_std.repeat(data.size(1),1).transpose(0,1)
    data = (data-temp_mean)/temp_std
    return  data
    
def compute_log2(data):
    temp_colsum = torch.sum(data,1)
    temp_mean_colsum = torch.mean(temp_colsum)
    temp = temp_colsum/temp_mean_colsum
    temp = temp.repeat(data.size(1),1).transpose(0,1)
    data = torch.log2(data/temp+1)
    return  data

def poe(mus, logvars, sample_mask):
    mu_stack  = torch.stack(mus, dim=1)
    logvar_stack = torch.stack(logvars, dim=1)

    precisions = torch.exp(-logvar_stack)          # [N, M, D]
    precisions = precisions * sample_mask.unsqueeze(-1).float()

    precision_sum = precisions.sum(dim=1) + 1e-8     # [N, D]
    weighted_mu_sum = (mu_stack * precisions).sum(dim=1)  # [N, D]
    pd_mu      = weighted_mu_sum / precision_sum
    pd_logvar  = torch.log(1.0 / precision_sum)
    return pd_mu, pd_logvar

def convert_to_longtensor(batch):
    batch = pd.Categorical(batch).codes
    batch = np.array(batch).astype('int32')
    batch = torch.from_numpy(batch)
    batch = batch.long()
    return batch

def process_highly_variable_genes(count_rna_tensor, n_top_genes=1000, target_sum=1e4):
    if n_top_genes!=None:
        adata = anndata.AnnData(count_rna_tensor.cpu().numpy())
        adata_new = adata.copy()
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
        hvg_mask = adata.var["highly_variable"]
    
        adata_hvg = adata_new[:, hvg_mask]
        count_rna_processed = torch.tensor(adata_hvg.X, dtype=torch.float32)
        return count_rna_processed, hvg_mask
    else:
        return count_rna_tensor, None
    
def get_ref_query_data(batch, condition, cty, count_rna, count_adt, count_atac, label, select_batch=[2, 5], keep_mask=None):
    # keep_mask, when given, selects cells directly (e.g. a donor-level held-out
    # set spanning batches); otherwise cells are selected by batch membership.
    if keep_mask is None:
        keep_mask = torch.zeros_like(batch, dtype=torch.bool)
        for q in select_batch:
            keep_mask |= (batch == q)

    batch = batch[keep_mask]
    condition = [c[keep_mask] for c in condition]

    cty = cty[keep_mask]
    if count_rna is not None:
        count_rna = count_rna[keep_mask, :]
    if count_adt is not None:
        count_adt = count_adt[keep_mask, :]
    if count_atac is not None:
        count_atac = count_atac[keep_mask, :]

    keep_mask_np = keep_mask.cpu().numpy()
    metadata_sub = label.iloc[keep_mask_np, :].copy()

    pd_batch = pd.Categorical(batch.cpu().numpy())
    batch = torch.tensor(pd_batch.codes, dtype=torch.long)
    return batch, condition, cty, count_rna, count_adt, count_atac, metadata_sub



def plot_umaps(data, metadata, color_keys, n_pcs: int = 30, n_neighbors: int = 100, random_state: int = 1234, point_size: int = 8):
    adata = anndata.AnnData(data, obs=metadata)
    sc.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
    sc.tl.umap(adata, random_state=random_state)
    for key in color_keys:
        if key not in adata.obs:
            print(f"⚠️  Warning: `{key}` not found in metadata; skip.")
            continue
        sc.pl.umap(adata, color=key, size=point_size, show=True)
    return adata 

def load_and_merge_metadata(metadata_path_list):
    metadata_list = []
    batch_vector = []

    for i, path in enumerate(metadata_path_list):
        if path is not None:
            df = pd.read_csv(path, header=0, index_col=False)
            metadata_list.append(df)
            batch_vector.extend([i] * len(df))

    if metadata_list:
        label = pd.concat(metadata_list, ignore_index=True)
        batch_tensor = torch.tensor(batch_vector)
        return label, batch_tensor
    else:
        return pd.DataFrame(), torch.tensor([])

def load_data_from_list(rna_path_list):
    rna_data_list = []
    for path in rna_path_list:
        if path is not None:
            rna_data = read_h5_data(path)
            rna_data = torch.transpose(rna_data, 1, 0)
            rna_data_list.append(rna_data)
        else:
            rna_data_list.append(None)
    return rna_data_list

def fill_missing_modalities(modality_lists):
    num_batches = len(modality_lists[0])
    batch_sizes = []
    for i in range(num_batches):
        size = None
        for modality in modality_lists:
            if modality[i] is not None:
                size = modality[i].shape[0]
                break
        if size is None:
            raise ValueError(f"Batch {i} has no available data in any modality to infer size.")
        batch_sizes.append(size)

    filled_modalities = []
    for modality in modality_lists:
        feature_dim = None
        for item in modality:
            if item is not None:
                feature_dim = item.shape[1]
                break
        if feature_dim is None:
            raise ValueError("Cannot infer feature dimension from empty modality.")

        filled = []
        for i, item in enumerate(modality):
            if item is None:
                filled.append(torch.zeros(batch_sizes[i], feature_dim))
            else:
                filled.append(item)
        filled_modalities.append(filled)
    return filled_modalities

def load_if_available(path_list):
    if any(p is not None for p in path_list):
        return load_data_from_list(path_list)
    else:
        return None

def fill_and_concat_available_lists(*modality_lists):
    available_lists = [lst for lst in modality_lists if lst is not None]
    
    if len(available_lists) == 0:
        return None, None, None

    if len(available_lists) == 1:
        filled = [torch.cat(available_lists[0], dim=0)]
        return (filled[0], None, None)

    num_batches = len(available_lists[0])
    batch_sizes = []
    for i in range(num_batches):
        for lst in available_lists:
            if lst[i] is not None:
                batch_sizes.append(lst[i].shape[0])
                break
        else:
            raise ValueError(f"Cannot infer batch size for batch {i}.")

    result_lists = []
    for full_list in modality_lists:
        if full_list is None:
            result_lists.append(None)
            continue

        feature_dim = next((x.shape[1] for x in full_list if x is not None), None)
        if feature_dim is None:
            raise ValueError("Cannot infer feature dimension from empty modality list.")

        filled = []
        for i in range(num_batches):
            if full_list[i] is None:
                filled.append(torch.zeros(batch_sizes[i], feature_dim))
            else:
                filled.append(full_list[i])
        result_lists.append(torch.cat(filled, dim=0))

    return tuple(result_lists)

def process_count_matrix(count, count_list, hvg_num):
    if count is not None:
        if hvg_num==None:
            return count
        if hvg_num!=None:
            count_valid = [x for x in count_list if x is not None]
            count_concat = torch.cat(count_valid, dim=0)
            nonzero_mask = count_concat.sum(dim=1) != 0
            count_nonzero = count_concat[nonzero_mask]
            count_temp, hvg_mask = process_highly_variable_genes(count_nonzero, hvg_num)
            count = count[:, hvg_mask]
            return count
    else:
        return None




def process_ref_count(ref_count, device, mask_poe_list, mask_recon_list, ref_data_list):
    #ref = compute_zscore(compute_log2(ref_count))
    ref = (compute_log2(ref_count))
    row_has_nonzero = (ref_count.abs().sum(dim=1) > 0).float()
    mask = row_has_nonzero.unsqueeze(1).repeat(1, ref_count.shape[1])
    mask_batch = torch.ones_like(ref_count) 

    mask_poe_list.append(mask[:, 1].unsqueeze(1).to(device))
    mask_recon_list.append(mask.to(device))
    ref_data_list.append(ref * mask)
    return mask_poe_list, mask_recon_list, ref_data_list, mask_batch

def process_ref_count_after_imputation(ref_count, device, mask_poe_list, mask_recon_list, ref_data_list):
    ref = ref_count #compute_zscore(compute_log2(ref_count))
    row_has_nonzero = (ref_count.abs().sum(dim=1) > 0).float()
    mask = row_has_nonzero.unsqueeze(1).repeat(1, ref_count.shape[1])
    mask_batch = torch.ones_like(ref_count) 

    mask_poe_list.append(mask[:, 1].unsqueeze(1).to(device))
    mask_recon_list.append(mask.to(device))
    ref_data_list.append(ref * mask)
    return mask_poe_list, mask_recon_list, ref_data_list, mask_batch



def KL_loss(mu, logvar):
    KLD = -0.5 * torch.mean(1 + logvar - mu**2 -  logvar.exp())
    return  KLD


class KL_loss(nn.Module):
    def __init__(self, beta=1.0, reduction='mean'):
        super(KL_loss, self).__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, mu, logvar):
        kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        
        if self.reduction == 'mean':
            kld = kld.mean()
        elif self.reduction == 'sum':
            kld = kld.sum()
        else:
            raise ValueError("reduction must be 'mean' or 'sum'")
        
        return self.beta * kld


def expand_encoder(seq: nn.Sequential,
                   new_query_batch_dim: int,
                   base_dim: int = 9,
                   init_fn = nn.init.xavier_uniform_,
                   freeze_base_cols: bool = True) -> nn.Sequential:
    old_linear = seq[0]
    if not isinstance(old_linear, nn.Linear):
        raise TypeError("seq[0] must be an nn.Linear")

    old_in  = old_linear.in_features
    out_f   = old_linear.out_features
    device  = old_linear.weight.device
    dtype   = old_linear.weight.dtype
    has_bias = old_linear.bias is not None

    if base_dim > old_in:
        raise ValueError(f"base_dim={base_dim} exceeds the old in_features={old_in}")

    new_in = base_dim + new_query_batch_dim

    new_linear = nn.Linear(new_in, out_f, bias=has_bias).to(device=device, dtype=dtype)
    with torch.no_grad():
        new_linear.weight[:, :base_dim].copy_(old_linear.weight[:, :base_dim])
        if new_in > base_dim:
            init_fn(new_linear.weight[:, base_dim:])
        if has_bias:
            new_linear.bias.copy_(old_linear.bias)

    if freeze_base_cols and new_in > base_dim:
        mask = torch.zeros_like(new_linear.weight)
        mask[:, base_dim:] = 1
        new_linear.weight.register_hook(lambda g, m=mask: g * m)

    seq[0] = new_linear
    return seq

def expand_decoder(module: nn.Module,
                               new_query_dim: int,
                               init_fn=nn.init.xavier_uniform_,
                               freeze_old_rows: bool = True) -> nn.Module:

    last_name, last_linear = None, None
    for name, m in module.named_modules():
        if isinstance(m, nn.Linear):
            last_name, last_linear = name, m
    if last_linear is None:
        raise ValueError("no nn.Linear found in the given module")

    old_out, old_in = last_linear.weight.shape
    new_out = old_out + int(new_query_dim)
    if new_out < old_out:
        raise ValueError(f"new_out({new_out}) must not be smaller than the old out_features({old_out})")

    device, dtype = last_linear.weight.device, last_linear.weight.dtype
    has_bias = last_linear.bias is not None

    new_linear = nn.Linear(old_in, new_out, bias=has_bias).to(device=device, dtype=dtype)
    with torch.no_grad():
        new_linear.weight[:old_out, :].copy_(last_linear.weight)
        if new_out > old_out:
            init_fn(new_linear.weight[old_out:, :])
        if has_bias:
            new_linear.bias[:old_out].copy_(last_linear.bias)
            if new_out > old_out:
                nn.init.zeros_(new_linear.bias[old_out:])

    if freeze_old_rows and new_out > old_out:
        wmask = torch.zeros_like(new_linear.weight)
        wmask[old_out:, :] = 1
        new_linear.weight.register_hook(lambda g, m=wmask: g * m)
        if has_bias:
            bmask = torch.zeros_like(new_linear.bias)
            bmask[old_out:] = 1
            new_linear.bias.register_hook(lambda g, m=bmask: g * m)

    parent = module
    leaf = last_name
    if "." in last_name:
        parent_path, leaf = last_name.rsplit(".", 1)
        parent = module.get_submodule(parent_path)
    parent._modules[leaf] = new_linear

    return module

def create_new_generator(generator, n_unique_ref_batch, n_unique_query_batch, n_unique_conditions):
    """Extend the batch dimension, freeze non-batch parameters, and build a new model."""
    new_generator = DeepMergeUncertaintyAware(
        n_features, 
        batch_classify_dim=n_unique_ref_batch + n_unique_query_batch,
        condition_dim=n_unique_conditions, z_dim=embedding_dim, hidden_features=[embedding_dim, embedding_dim]
    ).to("cuda")
    
    for name, param in generator.named_parameters():
        if "batch_encoder" not in name and "batch_decoder" not in name and "batch1_classify" not in name:
            new_param = dict(new_generator.named_parameters())[name]
            new_param.data.copy_(param.data)
            new_param.requires_grad = False

    new_generator.batch_encoder.encoder1 = expand_encoder(generator.batch_encoder.encoder1.net,  n_unique_query_batch, n_unique_ref_batch)
    new_generator.batch_encoder.encoder2 = expand_encoder(generator.batch_encoder.encoder2.net,  n_unique_query_batch, n_unique_ref_batch)

    new_generator.batch_decoder.decoder = expand_decoder(generator.batch_decoder.decoder.net,  n_unique_query_batch)

    new_generator.batch1_classify = expand_decoder(generator.batch1_classify.net,  n_unique_query_batch)

    return new_generator
