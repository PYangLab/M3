"""
End-to-end multi-resolution feature attribution for M3.

Connects the cell-level generator and patient-level corrector into one
differentiable forward pass, then runs Integrated Gradients from patient
disease prediction all the way back to raw gene expression.

Two levels of attribution:
1. Input-level IG: gene -> disease prediction (for gene and cell importance)
2. Patient vector-level IG: patient vector -> disease prediction (for cell type and donor importance)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from captum.attr import IntegratedGradients
from .util import poe


class EndToEndWrapper(nn.Module):

    def __init__(self, generator, corrector,
                 batch_labels, mask_poe,
                 donor_codes, celltype_codes,
                 n_celltypes, cond_index=0):
        super().__init__()
        self.generator = generator
        self.corrector = corrector

        self.register_buffer('batch_labels', batch_labels.long())
        self.register_buffer('mask_poe', mask_poe.float())
        self.register_buffer('donor_codes', donor_codes.long())
        self.register_buffer('celltype_codes', celltype_codes.long())

        self.n_celltypes = n_celltypes
        self.cond_index = cond_index
        self.n_conds = len(generator.condition_dim)
        self.batch_classify_dim = generator.batch_classify_dim
        self.n_cells = batch_labels.shape[0]

    def _get_z_slices(self, mu):
        z_dim = mu.shape[1]
        n_conds = self.n_conds
        emb_dim = z_dim - 2 - 2 * n_conds

        z_embedding = mu[:, :emb_dim]
        z_conds = []
        start = emb_dim
        for _ in range(n_conds):
            z_conds.append(mu[:, start:start + 2])
            start += 2

        return z_embedding, z_conds

    def _differentiable_aggregate(self, z, donor_codes, celltype_codes,
                                   n_celltypes, min_cells=5):
        feat_dim = z.shape[1]
        unique_donors = donor_codes.unique()
        n_donors = unique_donors.shape[0]

        patient_vectors = torch.zeros(n_donors, n_celltypes * feat_dim,
                                       device=z.device, dtype=z.dtype)

        for d_idx, d in enumerate(unique_donors):
            for c in range(n_celltypes):
                mask = (donor_codes == d) & (celltype_codes == c)
                if mask.sum() >= min_cells:
                    patient_vectors[d_idx, c * feat_dim:(c + 1) * feat_dim] = z[mask].mean(dim=0)

        return patient_vectors, unique_donors

    def _core_forward(self, x):
        mu_list, logvar_list = self.generator.encoder(x)
        b_onehot = F.one_hot(self.batch_labels,
                             num_classes=self.batch_classify_dim).float()
        batch_mu, batch_logvar = self.generator.batch_encoder(b_onehot)

        mu, logvar = poe(mu_list + [batch_mu],
                         logvar_list + [batch_logvar],
                         self.mask_poe)

        z_embedding, z_conds = self._get_z_slices(mu)
        z_for_agg = torch.cat([z_embedding, z_conds[self.cond_index]], dim=1)

        patient_vectors, unique_donors = self._differentiable_aggregate(
            z_for_agg, self.donor_codes, self.celltype_codes,
            self.n_celltypes
        )

        z_corrected = self.corrector.correct(patient_vectors)
        logits = self.corrector.status_head(z_corrected)
        return logits, patient_vectors, unique_donors

    def forward(self, x):
        logits, _, _ = self._core_forward(x)
        return logits

    def forward_flat(self, x_flat):
        x = x_flat.view(self.n_cells, -1)
        logits = self.forward(x)
        return logits.sum(dim=0, keepdim=True)

    def forward_full(self, x):
        logits, patient_vectors, unique_donors = self._core_forward(x)

        return {
            'logits': logits,
            'patient_vectors': patient_vectors,
            'unique_donors': unique_donors,
        }


def run_captum_ig(wrapper, x, target_class=1, n_steps=50, baseline=None):
    n_cells, n_genes = x.shape
    x_flat = x.detach().view(1, -1)
    if baseline is None:
        baseline_flat = torch.zeros_like(x_flat)
    else:
        baseline_flat = baseline.detach().reshape(1, -1)

    ig = IntegratedGradients(wrapper.forward_flat)

    attr_flat = ig.attribute(
        x_flat,
        baselines=baseline_flat,
        target=target_class,
        n_steps=n_steps,
        internal_batch_size=1,
    )

    attribution = attr_flat.view(n_cells, n_genes)
    return attribution


def run_attribution(generator, corrector,
                    ref_data, ref_b, ref_mask_poe,
                    ref_metadata, donor_name, cty_name,
                    n_celltypes, target_class=1,
                    cond_index=0, n_steps=50, device='cpu',
                    baseline=None, condition_col=None, reference_labels=None,
                    skip_patient_level=False):
    """
    Run end-to-end attribution on reference data using Captum IG.

    Two levels of IG:
    1. Input-level: gene expression -> disease prediction (for gene & cell importance)
    2. Patient vector-level: patient vector -> disease prediction (for cell type & donor importance)

    Args:
        condition_col: str, column name for condition in ref_metadata (e.g. 'cond_group').
        reference_labels: list of str, labels considered as reference/healthy 
                         (e.g. ['HC'] for COVID, ['Healthy'] for Stephenson).
        If condition_col and reference_labels are not provided, uses zero baseline 
        for patient vector level IG.
    """

    generator = generator.to(device).eval()
    corrector = corrector.to(device).eval()

    ref_data = ref_data.to(device).float()
    ref_b = ref_b.to(device)
    ref_mask_poe = ref_mask_poe.to(device)

    cty_categories = pd.Categorical(ref_metadata[cty_name])
    celltype_codes = torch.tensor(cty_categories.codes, dtype=torch.long, device=device)
    celltype_names = list(cty_categories.categories)

    donor_categories = pd.Categorical(ref_metadata[donor_name])
    donor_codes = torch.tensor(donor_categories.codes, dtype=torch.long, device=device)
    donor_names = list(donor_categories.categories)

    # build wrapper
    wrapper = EndToEndWrapper(
        generator, corrector,
        batch_labels=ref_b,
        mask_poe=ref_mask_poe,
        donor_codes=donor_codes,
        celltype_codes=celltype_codes,
        n_celltypes=n_celltypes,
        cond_index=cond_index
    ).to(device).eval()

    for p in wrapper.parameters():
        p.requires_grad_(False)

    # --- Patient-level info (no grad needed); skipped entirely when skip_patient_level ---
    if not skip_patient_level:
        print("Computing patient-level predictions...")
        with torch.no_grad():
            full_out = wrapper.forward_full(ref_data)
            logits = full_out['logits']
            patient_vectors = full_out['patient_vectors']
            unique_donors = full_out['unique_donors']
            disease_prob = F.softmax(logits, dim=1)[:, target_class]

    # =============================================
    # Level 1: Input-level IG (gene & cell importance)
    # =============================================
    print(f"Running Input-level Captum IG with {n_steps} steps...")
    if baseline is not None:
        baseline = baseline.to(device).float()
    attribution = run_captum_ig(
        wrapper, ref_data, target_class=target_class, n_steps=n_steps,
        baseline=baseline
    )
    print("Done.")

    # Gene importance and cell importance from input-level attribution
    attr_abs = attribution.abs()
    n_genes = attribution.shape[1]
    K = len(celltype_names)

    gene_celltype_matrix = torch.zeros(K, n_genes)
    for c in range(K):
        mask = (celltype_codes == c)
        if mask.sum() > 0:
            gene_celltype_matrix[c] = attribution[mask].mean(dim=0)

    gene_importance = attr_abs.mean(dim=0)          # absolute: gene involvement strength
    cell_importance = attribution.sum(dim=1)        # SIGNED: net per-cell push (+ -> target_class/Severe, - -> HC)

    # When patient-level attribution is skipped, derive cell-type importance from the
    # cell-level attribution (mean |cell push| per cell type) and return early — no
    # patient-vector IG, no per-donor outputs are computed at all.
    if skip_patient_level:
        celltype_importance = torch.zeros(K, device=cell_importance.device)
        for c in range(K):
            mask = (celltype_codes == c)
            if mask.any():
                celltype_importance[c] = cell_importance[mask].abs().mean()
        return {
            'attribution': attribution.detach().cpu(),
            'gene_celltype_matrix': gene_celltype_matrix.detach().cpu(),
            'gene_importance': gene_importance.detach().cpu(),
            'cell_importance': cell_importance.detach().cpu(),
            'celltype_importance': celltype_importance.detach().cpu(),
            'celltype_names': celltype_names,
        }

    # =============================================
    # Level 2: Patient vector-level IG (cell type & donor importance)
    # =============================================
    print(f"Running Patient vector-level Captum IG with {n_steps} steps...")

    def corrector_forward(pv):
        z = corrector.correct(pv)
        logits = corrector.status_head(z)
        return logits

    # Healthy baseline for patient vector
    if condition_col is not None and reference_labels is not None:
        donor_cond = ref_metadata.groupby(donor_name)[condition_col].first()
        healthy_donor_idx = [i for i, d in enumerate(donor_names)
                             if donor_cond.get(d, '') in reference_labels]
        if len(healthy_donor_idx) > 0:
            pv_baseline = patient_vectors[healthy_donor_idx].mean(dim=0, keepdim=True) \
                .expand_as(patient_vectors).contiguous()
        else:
            pv_baseline = torch.zeros_like(patient_vectors)
    else:
        pv_baseline = torch.zeros_like(patient_vectors)


    ig_patient = IntegratedGradients(corrector_forward)
    patient_level_attr = ig_patient.attribute(
        patient_vectors.detach().to(device),
        baselines=pv_baseline.detach().to(device),
        target=target_class,
        n_steps=n_steps
    )
    print("Done.")

    # Cell type importance from patient vector attribution
    feat_dim = patient_vectors.shape[1] // n_celltypes
    patient_attr_reshaped = patient_level_attr.view(-1, n_celltypes, feat_dim)
    celltype_importance = patient_attr_reshaped.abs().mean(dim=0).sum(dim=1)  # [K]

    # Donor importance from patient vector attribution
    donor_attribution = patient_level_attr.abs().sum(dim=1)  # [n_donors]

    return {
        # Input-level attribution (for gene & cell analysis)
        'attribution': attribution.detach().cpu(),
        'gene_celltype_matrix': gene_celltype_matrix.detach().cpu(),
        'gene_importance': gene_importance.detach().cpu(),
        'cell_importance': cell_importance.detach().cpu(),

        # Patient vector-level attribution (for cell type & donor analysis)
        'patient_level_attr': patient_level_attr.detach().cpu(),
        'celltype_importance': celltype_importance.detach().cpu(),
        'donor_attribution': donor_attribution.detach().cpu(),

        # Patient info
        'donor_disease_prob': disease_prob.detach().cpu(),
        'donor_names': donor_names,
        'patient_vectors': patient_vectors.detach().cpu(),
        'celltype_names': celltype_names,
    }
