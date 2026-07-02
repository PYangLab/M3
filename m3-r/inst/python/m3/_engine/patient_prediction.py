import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import os

import torch
import numpy as np
import random

def setup_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _as_tensor(x, device=None, dtype=torch.float32):
    if torch.is_tensor(x):
        t = x
    else:
        t = torch.tensor(x)
    t = t.to(dtype=dtype)
    if device is not None:
        t = t.to(device)
    return t

def _to_str_array(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().tolist()
    return np.array([str(v) for v in list(x)], dtype=object)

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.fc(x)


def encode_joint(ref, query):
    def to_list(x): return x.detach().cpu().tolist() if torch.is_tensor(x) else list(x)
    def norm(v):
        if isinstance(v, (int, np.integer)): return v
        return "Unknown" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
    r, q = list(map(norm, to_list(ref))), list(map(norm, to_list(query)))
    vocab = sorted(set(r + q))
    idx = {v:i for i,v in enumerate(vocab)}
    dev_r = ref.device if torch.is_tensor(ref) else torch.device('cpu')
    dev_q = query.device if torch.is_tensor(query) else torch.device('cpu')
    ref_codes  = torch.tensor([idx[v] for v in r], dtype=torch.long, device=dev_r)
    query_codes= torch.tensor([idx[v] for v in q], dtype=torch.long, device=dev_q)
    return ref_codes, query_codes, vocab

class LinBnDrop(nn.Sequential):
    def __init__(self, n_in, n_out, bn=True, p=0., act=None, lin_first=True):
        layers = [nn.BatchNorm1d(n_out if lin_first else n_in)] if bn else []
        if p != 0: layers.append(nn.Dropout(p))
        lin = [nn.Linear(n_in, n_out, bias=not bn)]
        if act is not None: lin.append(act)
        layers = lin + layers if lin_first else layers + lin
        super().__init__(*layers)

def make_mlp(d_in, d_h, d_out, p=0.1, bn=False):
    return nn.Sequential(
        LinBnDrop(d_in, d_h, bn=bn, p=p, act=nn.ReLU()),
        nn.Linear(d_h, d_out),
    )

class AdvSampleCorrector(nn.Module):
    def __init__(self, dim, n_con, n_batch, hidden=64, p=0.1, bn=False):
        super().__init__()
        self.transform = nn.Sequential(
            LinBnDrop(dim, hidden, bn=bn, p=p, act=nn.ReLU()),
            LinBnDrop(hidden, dim, bn=bn, p=0.0, act=None),
        )
        self.status_head = make_mlp(dim, hidden, n_con, p=p, bn=bn)
        self.disc = make_mlp(dim, hidden, n_batch, p=p, bn=bn)

    def correct(self, x):
        return self.transform(x)

    def forward(self, x):
        z = self.correct(x)
        return z, self.status_head(z), self.disc(z)
    

def aggregate_by_donor(z_embedding, donor, cty, min_cells=5, max_celltypes=None):
    unique_donors = torch.unique(donor)
    embedding_dim = z_embedding.shape[1]
    
    if max_celltypes is None:
        max_celltypes = int(cty.max().item()) + 1
    
    features = torch.zeros(len(unique_donors), max_celltypes * embedding_dim, device=z_embedding.device)
    
    for d_idx, d in enumerate(unique_donors):
        for c in range(max_celltypes):
            mask = (donor == d) & (cty == c)
            if mask.sum() >= min_cells:
                features[d_idx, c*embedding_dim:(c+1)*embedding_dim] = z_embedding[mask].mean(dim=0)
    
    return features, unique_donors

def process_dataset(z_embedding, donor, cty, condition,
                   min_cells_per_type=5, max_celltypes=None):
    
    features, unique_donors = aggregate_by_donor(
        z_embedding, donor, cty, min_cells_per_type, max_celltypes
    )
    
    donor_condition_map = {}
    for d in unique_donors:
        donor_mask = donor == d
        donor_condition_map[d.item()] = condition[donor_mask][0].item()
    
    labels = torch.tensor([donor_condition_map[d.item()] for d in unique_donors],
                         dtype=torch.long, device=z_embedding.device)
    
    return features.detach(), labels, unique_donors

def process_query_dataset(z_embedding, donor, cty,
                         min_cells_per_type=5, max_celltypes=None):
    
    features, unique_donors = aggregate_by_donor(
        z_embedding, donor, cty, min_cells_per_type, max_celltypes
    )
    
    return features.detach(), unique_donors

def fit_adv_sample_corrector(
    X_ref, y_con_ref, y_batch_ref,
    X_query, y_batch_query,
    n_con, n_batch,
    hidden=64, p=0.1, bn=False,
    lr=1e-3, weight_decay=1e-4,
    epochs=1000,
    n_disc=1,
    adv_warmup=50,
    adv_max=1.0,
    delta_l2=1e-3,
    batch_size=32,
    seed=2024,
    device=None,
    log_every=0,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    X_ref = X_ref.to(device).float()
    y_con_ref = y_con_ref.to(device).long()
    y_batch_ref = y_batch_ref.to(device).long()

    X_query = X_query.to(device).float()
    y_batch_query = y_batch_query.to(device).long()

    X_all = torch.cat([X_ref, X_query], dim=0)
    y_batch_all = torch.cat([y_batch_ref, y_batch_query], dim=0)

    y_con_all = torch.full((X_all.size(0),), -1, device=device, dtype=torch.long)
    y_con_all[:X_ref.size(0)] = y_con_ref

    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(X_all, y_batch_all, y_con_all),
        batch_size=min(batch_size, X_all.size(0)),
        shuffle=True,
        generator=g
    )

    model = AdvSampleCorrector(dim=X_all.size(1), n_con=n_con, n_batch=n_batch,
                               hidden=hidden, p=p, bn=bn).to(device)

    opt_disc = torch.optim.AdamW(model.disc.parameters(), lr=lr, weight_decay=weight_decay)
    opt_gen  = torch.optim.AdamW(
        list(model.transform.parameters()) + list(model.status_head.parameters()),
        lr=lr, weight_decay=weight_decay
    )

    ce_status = nn.CrossEntropyLoss()
    counts = torch.bincount(y_batch_all, minlength=n_batch).float()
    w = counts.sum() / (counts + 1e-8)
    w[counts == 0] = 0.0
    w = w / (w[counts > 0].mean() + 1e-8)
    w = w.to(device)
    ce_site = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)

    for ep in range(1, epochs + 1):
        model.train()
        adv_w = adv_max * min(1.0, ep / float(adv_warmup))
        first_batch = True
        for xb, yb_site, yb_status in loader:

            for _ in range(n_disc):
                opt_disc.zero_grad()
                with torch.no_grad():
                    z = model.correct(xb)          
                log_site = model.disc(z.detach())   
                loss_d = ce_site(log_site, yb_site)
                loss_d.backward()
                opt_disc.step()

            opt_gen.zero_grad()
            z = model.correct(xb)
            log_con = model.status_head(z)
            log_batch = model.disc(z)

            m = (yb_status >= 0)
            loss_s = ce_status(log_con[m], yb_status[m]) if m.any() else z.new_tensor(0.0)
            loss_adv = ce_site(log_batch, yb_site)

            loss = loss_s - adv_w * loss_adv
            loss.backward()
            opt_gen.step()
    return model


def donor_level_label(cell_donor_codes: torch.Tensor,
                      cell_site_codes: torch.Tensor,
                      unique_donors: torch.Tensor) -> torch.Tensor:
    out = torch.empty(len(unique_donors), dtype=torch.long, device=unique_donors.device)
    for i, d in enumerate(unique_donors):
        m = (cell_donor_codes == d)
        out[i] = cell_site_codes[m][0]  
    return out

def predict_with_reference_query(
    ref_z_embedding, ref_donor, ref_cty, ref_condition,
    query_z_embedding, query_donor, query_cty,ref_batch,query_batch,
    min_cells_per_type=5,  
     output_dim=2,
    bc_hidden=64, bc_p=0.1, bc_bn=False,
    bc_lr=1e-3, bc_weight_decay=1e-4,
    bc_epochs=1000, bc_n_disc=1,
    bc_adv_warmup=40, bc_adv_max=1.5,
    bc_batch_size=32,
    bc_seed=2024,do_batch_correction=True
):

    ref_cty, query_cty, cty_vocab = encode_joint(ref_cty, query_cty)
    ref_donor, query_donor, donor_vocab = encode_joint(ref_donor, query_donor)
    ref_condition, _, condition_vocab = encode_joint(ref_condition, ref_condition)
    max_celltypes = len(cty_vocab)

    ref_features, ref_labels, ref_donors = process_dataset(
        ref_z_embedding, ref_donor, ref_cty, ref_condition,
        min_cells_per_type, max_celltypes
    )

    query_features, query_donors = process_query_dataset(
        query_z_embedding, query_donor, query_cty,
        min_cells_per_type, max_celltypes
    )

    ref_features_bc = ref_features
    query_features_bc = query_features
    if torch.is_tensor(ref_batch):
        ref_batch = ref_batch.detach().cpu()
    if torch.is_tensor(query_batch):
        query_batch = query_batch.detach().cpu()

    ref_batch_codes, query_batch_codes, batch_vocab = encode_joint(ref_batch, query_batch)
    n_batch = len(batch_vocab)
    y_batch_ref = donor_level_label(ref_donor.cpu(), ref_batch_codes.cpu(), ref_donors.cpu()).to(ref_features.device)
    y_batch_qry = donor_level_label(query_donor.cpu(), query_batch_codes.cpu(), query_donors.cpu()).to(ref_features.device)
    n_con = int(ref_labels.max().item()) + 1

    if output_dim != n_con:
        output_dim = n_con  

    # device_bc = ref_features.device
    # device_bc = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_bc = torch.device("cpu")
    corrector = None
    if do_batch_correction:
        corrector = fit_adv_sample_corrector(
            X_ref=ref_features.to(device_bc), y_con_ref=ref_labels.to(device_bc), y_batch_ref=y_batch_ref.to(device_bc),
            X_query=query_features.to(device_bc), y_batch_query=y_batch_qry.to(device_bc),
            n_con=output_dim, n_batch=n_batch,
            hidden=bc_hidden, p=bc_p, bn=bc_bn,
            lr=bc_lr, weight_decay=bc_weight_decay,
            epochs=bc_epochs, n_disc=bc_n_disc,
            adv_warmup=bc_adv_warmup, adv_max=bc_adv_max,
            batch_size=bc_batch_size,
            seed=bc_seed, log_every=0,device=device_bc
        )

        corrector.eval()
        with torch.no_grad():
            ref_features_bc = corrector.correct(ref_features.to(device_bc)).to(ref_features.device)
            query_features_bc = corrector.correct(query_features.to(device_bc)).to(ref_features.device)
    else:
        ref_features_bc = ref_features
        query_features_bc = query_features


    query_donor_names = np.array([donor_vocab[int(i.item())] for i in query_donors], dtype=object)
    ref_donor_names = np.array([donor_vocab[int(i.item())] for i in ref_donors], dtype=object)

    with torch.no_grad():
        z_query = corrector.correct(query_features.to(device_bc))
        predictions = corrector.status_head(z_query)
        probabilities = F.softmax(predictions, dim=1)       
        pred_labels = torch.argmax(predictions, dim=1)

    return (
        pred_labels.cpu().numpy(),
        probabilities.cpu().numpy(),
        query_donor_names,
        query_features,        
        ref_features,         
        ref_labels,
        ref_donors,
        query_features_bc,    
        ref_features_bc,       
        ref_donor_names,
        condition_vocab        
    )
