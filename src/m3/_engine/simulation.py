import torch
import numpy as np
import pandas as pd

@torch.no_grad()
def synthesize_per_class(generator, ref_data, ref_metadata, ref_b, ref_mask_poe, celltype_name, per_class=500, tau=0.8,
                         device="cuda", mask=None, seed=42):
    torch_generator = torch.Generator(device=device).manual_seed(seed)
    generator.to(device).eval()
    celltypes = ref_metadata[celltype_name].to_numpy()
    batches = ref_b                              # shape [N]
    X = ref_data.to(device)                      # [N, F]
    B = torch.as_tensor(batches, device=device).long()
    m = ref_mask_poe.to(device) if mask is None else mask.to(device)

    *_, mu_all, logvar_all = generator(X, B, m)      # [N, zdim] x2
    std_all = (0.5 * logvar_all).exp()               # [N, zdim]

    if per_class == -1:
        eps = torch.randn(std_all.shape, device=std_all.device, generator=torch_generator)  # [N, zdim]
        z_all = mu_all + tau * std_all * eps
        x_gen_all = generator.decoder(z_all)             # [N, F]
        return x_gen_all.detach().cpu()
    else:
        out = {}
        classes = np.unique(celltypes)
        for cls in classes:
            idx_np = np.where(celltypes == cls)[0]
            if idx_np.size == 0:
                continue
            idx = torch.as_tensor(idx_np, device=device)
    
            sel = idx[torch.randint(0, idx.numel(), (per_class,), generator=torch_generator, device=device)]
            mu  = mu_all[sel]            # [per_class, zdim]
            std = std_all[sel]           # [per_class, zdim]
            eps = torch.randn(std.shape, device=std.device, generator=torch_generator)
            z   = mu + tau * std * eps
            x_gen = generator.decoder(z) # [per_class, F]
            out[str(cls)] = x_gen.detach().cpu()
            return out




@torch.no_grad()
def synthesize_donors_per_condition(
    generator,
    ref_data,
    ref_metadata,
    ref_b,
    ref_mask_poe,
    celltype_col,
    sample_col,
    condition_col,
    simulate_donor,
    num,
    batch_col=None,
    target_batch=-1,
    per_class=-1,
    tau=0.8,
    device="cuda",
    mask=None,
    seed=42,
):

    torch_generator = torch.Generator(device=device).manual_seed(seed)
    numpy_generator = np.random.default_rng(seed)   # donor-template selection (was global np.random)
    generator.to(device).eval()

    # Prepare tensors
    X = ref_data.to(device)
    B = torch.as_tensor(ref_b, device=device).long()
    M = ref_mask_poe.to(device) if mask is None else mask.to(device)

    # Encode all cells
    *_, mu_all, logvar_all = generator(X, B, M)
    std_all = (0.5 * logvar_all).exp()

    metadata = ref_metadata.copy()
    metadata["index"] = np.arange(len(metadata))

    results = {}

    for cond, n_donors in zip(simulate_donor, num):
        cond_cells = metadata[metadata[condition_col] == cond]

        if batch_col is not None and target_batch != -1:
            unique_batches = metadata[batch_col].unique().tolist()
            if 0 <= target_batch < len(unique_batches):
                selected_batch = unique_batches[target_batch]
                cond_cells = cond_cells[cond_cells[batch_col] == selected_batch]
            else:
                print(f"[WARN] target_batch={target_batch} out of range ({len(unique_batches)} batches available); skipping batch filter.")

        if cond_cells.empty:
            print(f"[WARN] No cells found for condition {cond} (batch={target_batch}), skip.")
            continue

        results[cond] = {}

        for donor_idx in range(n_donors):
            sample_template = numpy_generator.choice(cond_cells[sample_col].unique())
            donor_cells = cond_cells[cond_cells[sample_col] == sample_template]
            celltypes = donor_cells[celltype_col].unique()

            donor_result = {}

            for ct in celltypes:
                ct_cells = donor_cells[donor_cells[celltype_col] == ct]["index"].to_numpy()
                if len(ct_cells) == 0:
                    continue

                idx = torch.as_tensor(ct_cells, device=device)
                if per_class == -1:
                    sel = idx
                else:
                    sel = idx[torch.randint(0, idx.numel(), (per_class,), generator=torch_generator, device=device)]

                mu = mu_all[sel]
                std = std_all[sel]

                eps = torch.randn(std.shape, device=std.device, generator=torch_generator)
                z = mu + tau * std * eps
                x_gen = generator.decoder(z)

                donor_result[str(ct)] = x_gen.detach().cpu()

            results[cond][f"donor_{donor_idx+1}"] = donor_result

    return results

def summarise_generated_results(results):
    donor_summary = {}

    for cond, donors in results.items():
        donor_summary[cond] = {}
        for donor_name, ct_dict in donors.items():
            all_x, all_labels = [], []

            for ct, x_gen in ct_dict.items():
                all_x.append(x_gen)
                all_labels.extend([ct] * x_gen.shape[0])

            donor_mat = torch.cat(all_x, dim=0)
            donor_labels = np.array(all_labels)

            donor_summary[cond][donor_name] = {
                "X": donor_mat,
                "celltype": donor_labels
            }

    return donor_summary

def combine_simulated_donors(donor_summary):
    all_X = []
    all_meta = []

    for cond, donor_dict in donor_summary.items():
        for donor, info in donor_dict.items():
            X = info["X"]
            celltypes = info["celltype"]
            n_cells = len(celltypes)
    
            meta = pd.DataFrame({
                "cond_group": [cond] * n_cells,
                "mergedcelltype": celltypes,
                "donor": [donor] * n_cells,
                "simulate": [1] * n_cells
            })
    
            all_X.append(X)
            all_meta.append(meta)


    out = torch.cat(all_X, dim=0)
    meta_out = pd.concat(all_meta, ignore_index=True)
    return out, meta_out

