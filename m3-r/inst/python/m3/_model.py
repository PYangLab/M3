"""The user-facing M3 model object.

Thin orchestration over the engine (``m3._engine``). The engine couples
data-loading + model-build + training inside ``run_M3_update_*``, so the trained
generator is created during :meth:`M3.train` (not at construction). Construction
validates the column-role contract and freezes it on the object.

Capabilities:
  - Stage-1 integration VAE  -> embedding / reconstruct
  - Stage-2 donor predictor (when held_out + donor_key + celltype_key)
    -> predict_donors    (Stage-2 donor-level disease predictor)
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import tempfile
import threading
import uuid
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from m3 import _bridge

_MODALITY_ORDER = ("rna", "adt", "atac")

# obs column that carries each cell's Dataset row through the engine, so the rows
# the read-outs describe (a batch-grouped, possibly subsampled and permuted set)
# can be traced back to the Dataset. Removed again before train() returns.
_ROW = "__m3_row__"

# Bumped whenever the layout of a saved model file changes incompatibly.
_FORMAT = "m3-model"
_FORMAT_VERSION = 1


_QUIET = threading.Lock()


@contextlib.contextmanager
def _quiet_engine():
    """Silence the progress lines the engine's file loaders print.

    Done by shadowing ``print`` inside the loaders' own module rather than by
    redirecting ``sys.stdout``, so nothing the rest of the program prints -- from
    this thread or another -- is lost. The lock keeps two threads from undoing
    each other's shadowing.
    """
    from m3._engine import util
    with _QUIET:
        util.print = lambda *args, **kwargs: None
        try:
            yield
        finally:
            del util.print


class M3CapabilityError(RuntimeError):
    """Raised when a readout needs a capability the model has not been trained for."""


class _NotAModel(Exception):
    """Internal: why a file cannot be a model that save() wrote."""


class M3:
    def __init__(
        self,
        dataset,
        *,
        condition_keys,
        target_condition: str | None = None,
        donor_key: str | None = None,
        celltype_key: str | None = None,
        batch_key: str = "batch",
        held_out: list | None = None,
        held_out_samples: list | None = None,
        hvg: dict | None = None,
        embedding_dim: int = 30,
    ):
        if isinstance(condition_keys, str):
            condition_keys = [condition_keys]
        if not condition_keys:
            raise ValueError("condition_keys is required (m3 is condition-aware).")
        if len(condition_keys) > 2:
            raise ValueError(
                f"m3 supports at most 2 condition keys; got {len(condition_keys)} "
                f"({list(condition_keys)}). The engine's readout only returns results for "
                "1-2 conditions (3+ fails downstream after the whole training run).")
        for col in condition_keys:
            if col not in dataset.obs.columns:
                raise ValueError(f"condition key '{col}' not in dataset.obs.")
        for nm, col in (("celltype_key", celltype_key), ("donor_key", donor_key),
                        ("batch_key", batch_key)):
            if col is not None and col not in dataset.obs.columns:
                raise ValueError(f"{nm} '{col}' not in dataset.obs.")

        # The engine groups cells for integration by obs['batch']. Route batch_key
        # there (on a copy; the caller's Dataset is untouched) so the column the user
        # named is the one that is integrated, held out and validated against.
        if batch_key != "batch":
            clash = [k for k in (*condition_keys, donor_key, celltype_key) if k == "batch"]
            if clash:
                raise ValueError(
                    "obs column 'batch' is used as a role column and would be overwritten "
                    f"by batch_key='{batch_key}'; rename one of them.")
            import dataclasses
            dataset = dataclasses.replace(
                dataset, obs=dataset.obs.assign(batch=dataset.obs[batch_key]))
        if dataset.obs["batch"].nunique() == 1:
            warnings.warn(
                "only one batch found; no batch correction will be applied.", stacklevel=2)

        self.dataset = dataset
        self.condition_keys = list(condition_keys)
        self.target_condition = target_condition or self.condition_keys[0]
        if self.target_condition not in self.condition_keys:
            raise ValueError("target_condition must be one of condition_keys.")
        self._ci = self.condition_keys.index(self.target_condition)
        self.donor_key = donor_key
        self.celltype_key = celltype_key
        self.batch_key = batch_key
        self.site_col = batch_key  # column the donor adversary removes
        self.held_out = list(held_out) if held_out else []
        self.held_out_samples = list(held_out_samples) if held_out_samples else []
        if self.held_out and self.held_out_samples:
            raise ValueError("pass held_out (batches) OR held_out_samples (donors), not both.")
        if self.held_out_samples and not self.donor_key:
            raise ValueError("held_out_samples requires donor_key.")
        self.hvg = hvg or {}
        self.embedding_dim = embedding_dim

        batches = list(dataset.batches)
        for c in self.held_out:
            if c not in batches:
                raise ValueError(f"held_out batch '{c}' not among dataset batches {batches}.")
        if self.held_out_samples:
            _donors = set(dataset.obs[self.donor_key].astype(str))
            _missing = [s for s in map(str, self.held_out_samples) if s not in _donors]
            if _missing:
                raise ValueError(f"held_out_samples not in donor_key '{self.donor_key}': {_missing}.")

        # boolean mask of held-out (query) cells: by donor if held_out_samples, else by batch.
        if self.held_out_samples:
            self._query_cells = dataset.obs[self.donor_key].astype(str).isin(
                set(map(str, self.held_out_samples))).to_numpy()
        elif self.held_out:
            self._query_cells = dataset.obs[self.batch_key].isin(self.held_out).to_numpy()
        else:
            self._query_cells = np.zeros(dataset.obs.shape[0], dtype=bool)

        # NaN check on the SELECTED role columns only (not every obs column).
        # Held-out cells may legitimately have an unknown target_condition (the
        # model masks them), so condition NaN there is allowed and filled later.
        role_cols = list(dict.fromkeys(
            list(self.condition_keys)
            + [c for c in (self.donor_key, self.celltype_key, self.batch_key) if c]))
        for col in role_cols:
            if col not in dataset.obs.columns:
                continue
            vals = dataset.obs[col]
            if col in self.condition_keys:
                vals = vals[~self._query_cells]
            if vals.isna().any():
                warnings.warn(
                    f"role column '{col}' contains NaN; it will become a literal 'nan' category.",
                    stacklevel=2)

        self.contract = {
            "condition_keys": self.condition_keys,
            "target_condition": self.target_condition,
            "donor_key": self.donor_key,
            "celltype_key": self.celltype_key,
            "site_col": self.site_col,
            "held_out": self.held_out,
            "batches": batches,
            "modalities": dataset.modality_names,
            "hvg": self.hvg,
            "embedding_dim": self.embedding_dim,
            "reference_vocab": {
                k: sorted(map(str, dataset.obs.loc[~self._query_cells, k].dropna().unique()))
                for k in self.condition_keys
            },
        }

        self._generator = None
        self._corrector = None
        self._dp = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._all_data = self._all_b = self._all_mask_poe = None
        self._all_metadata = None
        self._all_rows = None
        self._hvg_cols = {}
        self._trained_on = None
        self._n_ref = 0
        self._n_conditions = len(self.condition_keys)
        self.history = None
        self.capabilities = {"embedding": False, "reconstruct": False, "predict_donors": False}

    # ------------------------------------------------------------------ train
    def train(
        self,
        *,
        max_epochs: int = 300,
        lr: float = 1e-5,
        batch_size: int = 256,
        early_stop_patience: int = 300,
        min_delta: float = 0.0,
        val_percentage: float = 0.1,
        weight_batch_ae: float = 1.0,
        weight_modality: dict | list | None = None,
        balance_batches: bool = True,
        donor_prediction: bool | None = None,
        donor_predictor: dict | None = None,
        seed: int | None = 0,
    ):
        """Train the complete model.

        With held_out + donor_key + celltype_key, trains the integration VAE
        leak-safe (query labels masked) and the donor predictor on top.
        Otherwise trains integration only.

        ``balance_batches=True`` (the default) trains the integration VAE on a
        batch-balanced subset; downstream steps (Stage-2 corrector, attribution,
        readouts) are then automatically rebuilt on the FULL reference set so
        sparse cell types aren't biased out.

        ``weight_modality`` weights each modality's reconstruction loss. Pass a
        dict keyed by modality name (e.g. ``{"rna": 1.0, "atac": 0.2}`` to
        down-weight the noisier ATAC) or a list in rna/adt/atac-present order;
        ``None`` (default) weights every present modality equally (1.0).

        ``seed`` (default 0) is applied right before the Stage-1 integration VAE,
        which the upstream engine otherwise leaves unseeded -- without it a run
        drifts run-to-run and is not guaranteed to match the R interface. Seeding
        here (the same way ``m3_train(seed=)`` does) makes a run reproducible and
        R==Python identical. Pass ``seed=None`` for the engine's unseeded behaviour.
        """
        from m3._engine.run_M3 import run_M3_update_no_classify, run_M3_update_with_query

        if self.celltype_key is None:
            raise ValueError("celltype_key is currently required for train().")

        want_donor = donor_prediction
        if want_donor is None:
            want_donor = bool(self.donor_key and self.celltype_key)
        if want_donor and not self.donor_key:
            raise ValueError("donor prediction / attribution needs donor_key + celltype_key.")

        # Start from a clean slate, so nothing of an earlier train() (a donor
        # predictor this run may not fit, say) outlives it.
        self._generator = self._corrector = self._dp = self.history = None
        self.capabilities = dict.fromkeys(self.capabilities, False)
        self._hvg_cols = {}
        self._trained_on = self._fingerprint()
        marshalled = _bridge.marshal(self._engine_dataset())
        mp = marshalled["modality_paths"]
        batches = marshalled["batches"]
        save_path = tempfile.mkdtemp(prefix="m3_train_")
        present = [m for m in _MODALITY_ORDER if m in self.dataset.modality_names]
        if weight_modality is None:
            weight_modality = [1.0] * len(present)
        elif isinstance(weight_modality, dict):
            weight_modality = [float(weight_modality.get(m, 1.0)) for m in present]
        else:
            weight_modality = [float(w) for w in weight_modality]
            if len(weight_modality) != len(present):
                raise ValueError(
                    f"weight_modality has {len(weight_modality)} entries but the dataset "
                    f"has modalities {present}; pass a dict keyed by modality name, or a "
                    f"list in that order.")
        hvg_num = [self.hvg.get(m) for m in _MODALITY_ORDER]

        held_idx = [batches.index(c) for c in self.held_out]
        train_idx = [i for i in range(len(batches)) if i not in held_idx]
        use_query = bool(held_idx) or bool(self.held_out_samples)
        select_test = held_idx if held_idx else None
        run_fn = run_M3_update_with_query if use_query else run_M3_update_no_classify
        # donor-level holdout: the engine builds the query mask from donor_name
        extra = (dict(held_out_samples=self.held_out_samples, donor_name=self.donor_key)
                 if self.held_out_samples else {})

        # The upstream engine leaves the Stage-1 integration VAE unseeded (run-to-run
        # drift); seed here -- mirroring the R interface -- so the run is reproducible
        # and R==Python identical. seed=None keeps the engine's unseeded behaviour.
        if seed is not None:
            from m3._engine.util import setup_seed
            setup_seed(seed)

        out = run_fn(
            mp["rna"], mp["adt"], mp["atac"], marshalled["metadata_paths"], save_path,
            self.condition_keys, self.celltype_key, batch_size, lr, max_epochs,
            min_delta, early_stop_patience, val_percentage, hvg_num,
            weight_modality, weight_batch_ae, self.embedding_dim,
            train_idx, select_test, balance_batches, **extra,
        )
        # The generator was trained on the (possibly balanced) subset. The two
        # downstream paths handle the data differently:
        #   - NO held-out query (attribution / feature selection): when balanced,
        #     re-assemble the FULL reference set for Stage-2 + attribution.
        #   - WITH a held-out query (patient prediction): do NOT re-assemble; Stage-2
        #     runs on the balanced reference returned by Stage-1.
        # Applying the feature_selection re-assembly to the patient-prediction path
        # changes the ref-donor features and shifts borderline query predictions.
        if balance_batches and not use_query:
            full_out = run_fn(
                mp["rna"], mp["adt"], mp["atac"], marshalled["metadata_paths"], save_path,
                self.condition_keys, self.celltype_key, batch_size, lr, 0,
                min_delta, early_stop_patience, val_percentage, hvg_num,
                weight_modality, weight_batch_ae, self.embedding_dim,
                train_idx, select_test, False,
            )
            (ref_data, ref_b, ref_mask_poe, ref_metadata,
             query_data, query_b, query_mask_poe, query_metadata, _gen_full,
             _p0, _p1) = full_out
            generator = out[8]  # keep the BALANCED-trained weights
        else:
            (ref_data, ref_b, ref_mask_poe, ref_metadata,
             query_data, query_b, query_mask_poe, query_metadata, generator, _p0, _p1) = out

        self._generator = generator.to(self._device)
        self._n_ref = int(ref_data.shape[0])
        if query_data is not None and getattr(query_data, "shape", [0])[0] > 0:
            self._all_data = torch.cat([ref_data, query_data], dim=0)
            self._all_b = torch.cat([ref_b.reshape(-1), query_b.reshape(-1)], dim=0)
            self._all_mask_poe = torch.cat([ref_mask_poe, query_mask_poe], dim=0)
            self._all_metadata = pd.concat(
                [ref_metadata.reset_index(drop=True), query_metadata.reset_index(drop=True)],
                ignore_index=True)
        else:
            self._all_data, self._all_b, self._all_mask_poe = ref_data, ref_b, ref_mask_poe
            self._all_metadata = ref_metadata.reset_index(drop=True)

        self._all_rows = self._all_metadata.pop(_ROW).to_numpy(dtype=np.int64)
        self._all_data = self._all_data.to(self._device)
        self._all_b = self._all_b.to(self._device).long()
        self._all_mask_poe = self._all_mask_poe.to(self._device)
        shutil.rmtree(marshalled["tmpdir"], ignore_errors=True)
        shutil.rmtree(save_path, ignore_errors=True)

        self.capabilities["embedding"] = True
        self.capabilities["reconstruct"] = True

        if want_donor:
            self._fit_donor_predictor(donor_predictor or {})
            self.capabilities["predict_donors"] = True
        return self

    # ------------------------------------------------- engine-side data
    def _engine_dataset(self):
        """The Dataset as it is handed to the engine: held-out cells' missing
        condition labels filled in, and every cell tagged with its row."""
        import dataclasses
        obs = self.dataset.obs
        if _ROW in obs.columns:
            raise ValueError(f"obs column '{_ROW}' is reserved by m3; rename it.")
        if self._query_cells.any() and any(
                obs.loc[self._query_cells, k].isna().any() for k in self.condition_keys):
            # Held-out cells' condition labels are masked in training anyway; fill
            # any NaN with a reference placeholder so the engine sees a valid
            # category (no requirement to supply query labels).
            obs = obs.copy()
            for k in self.condition_keys:
                ph = self.contract["reference_vocab"][k][0]
                col = obs[k].astype(object)
                col[self._query_cells & obs[k].isna().to_numpy()] = ph
                obs[k] = col
        return dataclasses.replace(
            self.dataset, obs=obs.assign(**{_ROW: np.arange(obs.shape[0])}))

    def _assemble(self, rows, n_ref: int):
        """Rebuild the read-out tensors and metadata for Dataset rows ``rows``
        (the first ``n_ref`` are the reference set, the rest the held-out query).

        train() gets these from the engine run; a loaded model has no run to take
        them from, so this repeats the engine's own steps -- same file round trip,
        same loaders, same per-set normalisation -- on the rows the saved model
        was read out on. save() checks that it reproduces the trained model's
        tensors exactly before it writes anything.
        """
        from m3._engine import util as U

        marshalled = _bridge.marshal(self._engine_dataset())
        try:
            mp = marshalled["modality_paths"]
            with _quiet_engine():
                label, batch = U.load_and_merge_metadata(marshalled["metadata_paths"])
                counts = list(U.fill_and_concat_available_lists(
                    *[U.load_if_available(mp[m]) for m in _MODALITY_ORDER]))
        finally:
            shutil.rmtree(marshalled["tmpdir"], ignore_errors=True)
        for slot, m in enumerate(_MODALITY_ORDER):
            if m in self._hvg_cols and counts[slot] is not None:
                counts[slot] = counts[slot][:, torch.as_tensor(self._hvg_cols[m])]

        # engine order (batch after batch) -> position of each requested row
        pos = np.empty(len(label), dtype=np.int64)
        pos[label[_ROW].to_numpy()] = np.arange(len(label))
        sel = pos[np.asarray(rows, dtype=np.int64)]
        dev = self._device

        def block(idx):
            idx = torch.as_tensor(idx)
            mask_poe, mask_recon, data = [], [], []
            for c in counts:
                if c is not None:
                    mask_poe, mask_recon, data, mask_batch = U.process_ref_count(
                        c[idx], dev, mask_poe, mask_recon, data)
            data = [torch.nan_to_num(d, nan=0.0) for d in data]
            codes = pd.Categorical(batch[idx].cpu().numpy()).codes
            return (torch.cat(data, dim=1),
                    torch.cat(mask_poe + [mask_batch[:, 1].unsqueeze(1).to(dev)], dim=1),
                    torch.tensor(codes, dtype=torch.long))

        data, mask, b = block(sel[:n_ref])
        if n_ref < len(sel):
            q_data, q_mask, q_b = block(sel[n_ref:])
            q_b = q_b + max(b) + 1
            data = torch.cat([data, q_data], dim=0)
            mask = torch.cat([mask, q_mask], dim=0)
            b = torch.cat([b.reshape(-1), q_b.reshape(-1)], dim=0)
        metadata = label.iloc[sel].reset_index(drop=True).drop(columns=_ROW)
        return data.to(dev), b.to(dev).long(), mask.to(dev), metadata

    # ---------------------------------------------- Stage-2 donor predictor
    def _donor_setup(self, min_cells: int):
        """Donor, cell-type, condition and batch codes of the read-out rows.

        Everything here follows from the rows' metadata, so it is rebuilt -- not
        stored -- when a model is loaded.
        """
        dev = self._device
        gen = self._generator
        ci = self._ci
        meta = self._all_metadata
        n_ref, n_total = self._n_ref, self._all_data.shape[0]

        gen.eval()
        with torch.no_grad():
            probe = gen(self._all_data[: min(64, n_total)], self._all_b[: min(64, n_total)],
                        self._all_mask_poe[: min(64, n_total)])
            emb_dim = int(probe[2].shape[1] + probe[4][ci].shape[1])
        self._emb_dim = emb_dim

        celltype_cat = pd.Categorical(meta[self.celltype_key].astype(str))
        n_celltypes = len(celltype_cat.categories)
        celltype_codes = torch.tensor(celltype_cat.codes.astype("int64").copy()).to(dev)
        donor_cat = pd.Categorical(meta[self.donor_key].astype(str))
        donor_codes = torch.tensor(donor_cat.codes.astype("int64").copy()).to(dev)

        # NOTE (assumption): cond_codes below are reference-only, string-sorted condition
        # codes. They align with the engine's cell-level condition classifier go[7][ci]
        # only when target_condition is a string label whose reference categories share the
        # engine's full-set ordering -- true for the shipped setup (string disease labels,
        # both classes present in the reference). For numeric-coded conditions or a
        # query-exclusive category the code spaces can diverge for the auxiliary cond_loss.
        ref_meta = meta.iloc[:n_ref]
        ref_cond_cat = pd.Categorical(ref_meta[self.target_condition].astype(str))
        condition_vocab = [str(c) for c in ref_cond_cat.categories]
        code_of = {v: i for i, v in enumerate(condition_vocab)}
        cond_codes = torch.tensor(
            [code_of.get(s, -1) for s in meta[self.target_condition].astype(str)],
            dtype=torch.long).to(dev)

        site_cat = pd.Categorical(meta[self.site_col].astype(str))
        site_codes_cell = torch.tensor(site_cat.codes.astype("int64").copy()).to(dev)
        is_ref = torch.zeros(n_total, dtype=torch.bool, device=dev)
        is_ref[:n_ref] = True

        donor_status, donor_site, donor_is_ref = {}, {}, {}
        for dc in torch.unique(donor_codes).tolist():
            m = (donor_codes == dc)
            donor_status[dc] = int(cond_codes[m][0])
            donor_site[dc] = int(site_codes_cell[m][0])
            donor_is_ref[dc] = bool(is_ref[m][0])

        self._dp = {
            "donor_cat": donor_cat, "condition_vocab": condition_vocab,
            "donor_codes": donor_codes, "celltype_codes": celltype_codes,
            "cond_codes": cond_codes,
            "donor_is_ref": donor_is_ref, "donor_status": donor_status,
            "donor_site": donor_site, "n_site": len(site_cat.categories),
            "n_celltypes": n_celltypes, "min_cells": min_cells,
        }

    def _agg_diff(self, z):
        """Per-donor feature: the mean latent of each cell type, side by side (zeros
        for a cell type the donor has fewer than ``min_cells`` cells of)."""
        dp = self._dp
        donor_codes, celltype_codes = dp["donor_codes"], dp["celltype_codes"]
        uds = torch.unique(donor_codes)
        rows = []
        for dc in uds:
            parts = [z[(donor_codes == dc) & (celltype_codes == k)].mean(0)
                     if int(((donor_codes == dc) & (celltype_codes == k)).sum()) >= dp["min_cells"]
                     else z.new_zeros(self._emb_dim)
                     for k in range(dp["n_celltypes"])]
            rows.append(torch.cat(parts))
        return torch.stack(rows), uds

    def _fit_donor_predictor(self, cfg: dict):
        """Stage-2 donor predictor: joint co-adaptation of encoder + corrector."""
        from m3._engine import patient_prediction as pp
        from m3._engine.util import setup_seed

        enc_lr = cfg.get("enc_lr", 1e-5)
        glr = cfg.get("glr", 5e-3)
        n_epochs = cfg.get("n_epochs", 30)
        patient_w = cfg.get("patient_w", 3.0)
        adv_max = cfg.get("adv_max", 1.0)
        adv_warmup = cfg.get("adv_warmup", 10)
        n_disc = cfg.get("n_disc", 1)
        hidden = cfg.get("hidden", 64)
        min_cells = cfg.get("min_cells", 5)

        self._donor_setup(min_cells)
        dev = self._device
        gen = self._generator
        ci = self._ci
        n_ref = self._n_ref
        emb_dim = self._emb_dim
        dp = self._dp
        n_celltypes, n_site = dp["n_celltypes"], dp["n_site"]
        n_cond_classes = len(dp["condition_vocab"])
        cond_codes = dp["cond_codes"]
        donor_status, donor_site, donor_is_ref = (
            dp["donor_status"], dp["donor_site"], dp["donor_is_ref"])
        agg_diff = self._agg_diff

        setup_seed(0)
        corrector = pp.AdvSampleCorrector(dim=n_celltypes * emb_dim, n_con=n_cond_classes,
                                          n_batch=n_site, hidden=hidden, p=0.1, bn=False).to(dev)
        opt_enc = torch.optim.AdamW(gen.encoder.parameters(), lr=enc_lr, weight_decay=1e-2)
        opt_cell = torch.optim.AdamW(
            list(gen.decoder.parameters()) + list(gen.condition_classifiers.parameters())
            + list(gen.batch_decoder.parameters()) + list(gen.batch_encoder.parameters())
            + list(gen.cty1_classify.parameters()), lr=enc_lr, weight_decay=1e-2)
        opt_gen = torch.optim.AdamW(
            list(corrector.transform.parameters()) + list(corrector.status_head.parameters()),
            lr=glr, weight_decay=1e-4)
        opt_disc = torch.optim.AdamW(corrector.disc.parameters(), lr=glr, weight_decay=1e-4)
        ce = nn.CrossEntropyLoss()
        mse_none = nn.MSELoss(reduction="none")
        mse_mean = nn.MSELoss()

        hist = []
        for epoch in range(1, n_epochs + 1):
            gen.train()
            go = gen(self._all_data, self._all_b, self._all_mask_poe)
            z = torch.cat([go[2], go[4][ci]], 1)
            recon_loss = mse_none(self._all_data, go[0]).mean()
            batch_oh = F.one_hot(self._all_b, num_classes=gen.batch_classify_dim).float()
            batch_recon = mse_mean(batch_oh, go[1])
            kl = (-0.5 * torch.sum(1 + go[9] - go[8].pow(2) - go[9].exp())) / self._all_data.shape[0]
            cond_loss = ce(go[7][ci][:n_ref], cond_codes[:n_ref])
            cell_loss = recon_loss + 1.0 * batch_recon + 0.0001 * kl + cond_loss
            donor_feat, uds = agg_diff(z)
            y_site = torch.tensor([donor_site[int(d)] for d in uds.tolist()], device=dev)
            y_status = torch.tensor([donor_status[int(d)] if donor_is_ref[int(d)] else -1
                                     for d in uds.tolist()], device=dev)
            for _ in range(n_disc):
                opt_disc.zero_grad()
                cdet = corrector.correct(donor_feat.detach())
                disc_loss = ce(corrector.disc(cdet.detach()), y_site)
                disc_loss.backward()
                opt_disc.step()
            adv_w = adv_max * min(1.0, epoch / float(adv_warmup))
            corr_grad = corrector.correct(donor_feat)
            mask = (y_status >= 0)
            status_loss = ce(corrector.status_head(corr_grad)[mask], y_status[mask])
            adv_loss = ce(corrector.disc(corr_grad), y_site)
            total = cell_loss + patient_w * (status_loss - adv_w * adv_loss)
            opt_enc.zero_grad()
            opt_cell.zero_grad()
            opt_gen.zero_grad()
            total.backward()
            opt_enc.step()
            opt_cell.step()
            opt_gen.step()
            hist.append({"epoch": epoch, "cell_loss": float(cell_loss.detach()),
                         "status_loss": float(status_loss.detach()), "adv_w": float(adv_w)})

        self._corrector = corrector
        self.history = hist
        dp["hidden"] = hidden

    def _build_donor_features(self):
        gen, corr, ci = self._generator, self._corrector, self._ci
        gen.eval()
        corr.eval()
        with torch.no_grad():
            go = gen(self._all_data, self._all_b, self._all_mask_poe)
            intrinsic = go[2].shape[1]
            mu = go[8]
            z = torch.cat([mu[:, :intrinsic], mu[:, intrinsic + 2 * ci: intrinsic + 2 * ci + 2]], 1)
            z = torch.where(torch.isnan(z) | torch.isinf(z), torch.zeros_like(z), z)
            donor_feat, uds = self._agg_diff(z)
            donor_corr = corr.correct(donor_feat)
            logits = corr.status_head(donor_corr)
        return donor_corr, logits, uds

    def predict_donors(self, *, include_reference: bool = False) -> pd.DataFrame:
        """Donor-level disease prediction (held-out query donors by default)."""
        if not self.capabilities["predict_donors"]:
            raise M3CapabilityError(
                "predict_donors requires a donor predictor; construct with held_out + "
                "donor_key + celltype_key and call train().")
        donor_corr, logits, uds = self._build_donor_features()
        prob = F.softmax(logits, 1).cpu().numpy()
        pred = torch.argmax(logits, 1).cpu().numpy()
        dp = self._dp
        names = [str(dp["donor_cat"].categories[int(d)]) for d in uds.tolist()]
        is_query = np.array([not dp["donor_is_ref"][int(d)] for d in uds.tolist()])
        vocab = dp["condition_vocab"]
        rows = {self.donor_key: names, "is_reference": ~is_query,
                "predicted_label": [vocab[i] for i in pred]}
        for i, v in enumerate(vocab):
            rows[f"prob_{v}"] = prob[:, i]
        df = pd.DataFrame(rows)
        if not include_reference:
            df = df[~df["is_reference"]].reset_index(drop=True)
            if df.empty:
                warnings.warn(
                    "predict_donors() has no query donors to report: the model was trained "
                    "without a held-out set (no held_out / held_out_samples), so every donor "
                    "is a reference donor. Pass include_reference=True to see reference-donor "
                    "predictions, or construct M3 with held_out=/held_out_samples= to define a "
                    "query set.", stacklevel=2)
        return df

    def donor_embedding(self) -> pd.DataFrame:
        """Patient/donor-level embedding -- the corrector-corrected donor vector used for
        prediction. Returns a DataFrame indexed by donor, with an `is_reference` column
        followed by the embedding dimensions (`m3_0`, `m3_1`, ...). Join with
        `predict_donors(include_reference=True)` for predicted labels.
        """
        if not self.capabilities["predict_donors"]:
            raise M3CapabilityError(
                "donor_embedding requires a donor predictor; construct with "
                "donor_key + celltype_key and call train().")
        donor_corr, _logits, uds = self._build_donor_features()
        emb = donor_corr.detach().cpu().numpy()
        dp = self._dp
        names = [str(dp["donor_cat"].categories[int(d)]) for d in uds.tolist()]
        is_ref = [bool(dp["donor_is_ref"][int(d)]) for d in uds.tolist()]
        df = pd.DataFrame(emb, index=pd.Index(names, name=self.donor_key),
                          columns=[f"m3_{i}" for i in range(emb.shape[1])])
        df.insert(0, "is_reference", is_ref)
        return df

    # ------------------------------------------------------------- readouts
    def _forward_mu(self):
        if self._generator is None:
            raise M3CapabilityError("model is not trained; call train() first.")
        self._generator.eval()
        with torch.no_grad():
            out = self._generator(self._all_data, self._all_b, self._all_mask_poe)
        return out[8], out[2].shape[1]

    def embedding(self, part: str = "bio") -> np.ndarray:
        """Cell-level latent. part in {'bio','intrinsic','batch', or a condition key}."""
        if not self.capabilities["embedding"]:
            raise M3CapabilityError("embedding requires train(); call train() first.")
        mu, intrinsic = self._forward_mu()
        nc = self._n_conditions
        if part == "intrinsic":
            emb = mu[:, :intrinsic]
        elif part == "batch":
            emb = mu[:, -2:]
        elif part == "bio":
            emb = mu[:, : intrinsic + 2 * nc]
        elif part in self.condition_keys:
            i = self.condition_keys.index(part)
            emb = mu[:, intrinsic + 2 * i: intrinsic + 2 * i + 2]
        else:
            raise ValueError(
                f"unknown part '{part}'; use 'bio'/'intrinsic'/'batch' or one of {self.condition_keys}.")
        return emb.detach().cpu().numpy()

    def reconstruct(self, *, remove_batch: bool = True) -> dict:
        """Batch-corrected per-modality reconstruction (posterior mean decode)."""
        if not self.capabilities["reconstruct"]:
            raise M3CapabilityError("reconstruct requires train(); call train() first.")
        mu, _ = self._forward_mu()
        mu_use = mu.clone()
        if remove_batch:
            mu_use[:, -2:] = 0.0
        with torch.no_grad():
            recon = self._generator.decoder(mu_use).detach().cpu().numpy()
        splits = list(self._generator.encoder.feature_splits)
        present = [m for m in _MODALITY_ORDER if m in self.dataset.modality_names]
        out, start = {}, 0
        for m, w in zip(present, splits):
            out[m] = recon[:, start: start + w]
            start += w
        return out

    def _hvg_index(self, mod: str, w: int, full: list) -> list:
        """Original-column indices of the ``w`` features the trained model uses for ``mod``.

        When in-model HVG selection is active (``hvg={mod: n}`` at construction), the
        engine keeps a *scattered* scanpy-selected subset of the original columns, not
        the first ``w``. Reproduce that selection with the engine's own (deterministic)
        ``process_highly_variable_genes`` so attribution feature names line up with the
        columns actually attributed. Returns ``range(w)`` when no in-model HVG was applied
        (full matrix -> the first ``w`` are all of them, so the prefix is already correct).
        """
        hvg_n = self.hvg.get(mod)
        if not hvg_n or w == len(full):
            return list(range(w))
        if mod in self._hvg_cols:        # recorded by save() / restored by load()
            return list(self._hvg_cols[mod])
        from m3._engine.util import process_highly_variable_genes
        mat = self.dataset.modalities[mod]                    # raw counts, all cells
        dense = torch.as_tensor(np.asarray(mat.todense()), dtype=torch.float32)
        dense = dense[dense.sum(dim=1) != 0]                  # engine drops all-zero cells first
        _, hvg_mask = process_highly_variable_genes(dense, int(hvg_n))
        idx = np.where(np.asarray(hvg_mask))[0].tolist()
        if len(idx) != w:
            raise RuntimeError(
                f"attribution could not map HVG feature names for modality {mod!r}: "
                f"reproduced {len(idx)} selected features but the model uses {w}. "
                "Pre-select HVGs before building the Dataset to avoid in-model hvg=.")
        return idx

    # ---------------------------------------------------------- attribution
    def attribute(self, *, reference_labels, target_class: int | None = None, n_steps: int = 50):
        """End-to-end integrated-gradients attribution.

        reference_labels: the healthy/baseline label(s) of target_condition (e.g. ['HC']).
            These are used to construct **two** independent IG baselines:
              - cell-level: mean expression vector across all cells with one of
                ``reference_labels`` in ``target_condition`` (so per-cell IG measures
                "how does this cell differ from the average HC cell");
              - patient-level: mean patient-vector over reference donors (inside
                ``run_attribution``, controls the donor-vector IG).

            Without this match-driver behaviour, the cell-level IG silently uses
            a zero baseline, which inflates pan-celltype / housekeeping genes.

        Returns an Attribution with .genes / .celltypes / .donors / .cells.
        """
        if not self.capabilities["predict_donors"]:
            raise M3CapabilityError(
                "attribute requires the donor predictor (corrector); construct with "
                "donor_key + celltype_key and call train().")
        from m3._engine.feature_selection import run_attribution

        vocab = self._dp["condition_vocab"]
        ref_set = set(map(str, reference_labels))
        if target_class is None:
            cand = [i for i, v in enumerate(vocab) if v not in ref_set]
            if not cand:
                raise ValueError(f"no non-reference class in vocab {vocab} vs {sorted(ref_set)}.")
            target_class = cand[0]

        # Cell-level IG baseline: mean expression of reference (e.g. HC) cells.
        # Without this the engine
        # falls back to a zero baseline, which makes housekeeping / pan-celltype
        # genes (FOS, ACTB, HSPA8, ...) dominate the ranking because their
        # contribution from 0 -> observed value is trivially large.
        cond_str = self._all_metadata[self.target_condition].astype(str).to_numpy()
        ref_mask = np.isin(cond_str, [str(x) for x in reference_labels])
        ref_data_cpu = self._all_data.detach().cpu().float()
        if ref_mask.any():
            ref_mean = ref_data_cpu[torch.as_tensor(ref_mask)].mean(0, keepdim=True)
            cell_baseline = ref_mean.expand_as(ref_data_cpu).contiguous()
        else:
            warnings.warn(
                f"no cells with {self.target_condition} in {list(reference_labels)} "
                "found; falling back to zero baseline.", stacklevel=2)
            cell_baseline = torch.zeros_like(ref_data_cpu)

        res = run_attribution(
            self._generator, self._corrector,
            self._all_data, self._all_b, self._all_mask_poe,
            self._all_metadata, self.donor_key, self.celltype_key,
            self._dp["n_celltypes"], target_class=target_class, cond_index=self._ci,
            n_steps=n_steps, device=str(self._device),
            baseline=cell_baseline, condition_col=self.target_condition,
            reference_labels=list(reference_labels))

        present = [m for m in _MODALITY_ORDER if m in self.dataset.modality_names]
        # Per-modality slices use the POST-HVG widths of the trained generator
        # (the engine's feature_splits), not the raw var lists.
        splits = list(self._generator.encoder.feature_splits)
        feat_names: list[str] = []
        modality_of: list[str] = []
        for mod, w in zip(present, splits):
            full = list(self.dataset.var[mod])
            idx = self._hvg_index(mod, w, full)  # model columns -> original var positions
            feat_names.extend(full[i] for i in idx)
            modality_of.extend([mod] * w)
        return Attribution(
            res, feat_names, target_label=vocab[target_class],
            modality_of=modality_of,
            cell_metadata=self._all_metadata,
            celltype_key=self.celltype_key,
            donor_key=self.donor_key,
            target_condition=self.target_condition,
            reference_labels=list(reference_labels))

    # ---------------------------------------------------------- generation
    def generate(self, *, tau: float = 0.8, seed: int = 42) -> dict:
        """Posterior-resampled synthetic cells (1:1 with reference), per modality."""
        if not self.capabilities["embedding"]:
            raise M3CapabilityError("generate requires train(); call train() first.")
        from m3._engine.simulation import synthesize_per_class
        x = synthesize_per_class(
            self._generator, self._all_data, self._all_metadata, self._all_b,
            self._all_mask_poe, self.celltype_key, per_class=-1, tau=tau,
            device=str(self._device), seed=seed)
        arr = x.numpy() if hasattr(x, "numpy") else np.asarray(x)
        splits = list(self._generator.encoder.feature_splits)
        present = [m for m in _MODALITY_ORDER if m in self.dataset.modality_names]
        out, start = {}, 0
        for m, w in zip(present, splits):
            out[m] = arr[:, start: start + w]
            start += w
        return out

    def augment(self, *, conditions, n_donors, tau: float = 0.8,
                batch: str | None = None, seed: int = 42) -> dict:
        """Synthesize new donors per condition by posterior-resampling real donor templates.

        batch: optional batch label (looked up against the column passed as
            ``batch_key`` at construction time). When set, template samples for
            generation are restricted to that batch -- useful for reproducing
            batch-stratified figures.

        Returns {'expression': {modality: array}, 'obs': DataFrame}.
        """
        if not self.capabilities["embedding"]:
            raise M3CapabilityError("augment requires train(); call train() first.")
        if self.donor_key is None:
            raise M3CapabilityError(
                "augment requires donor_key at construction (it resamples donor templates).")
        conditions, n_donors = list(conditions), list(n_donors)
        if len(conditions) != len(n_donors):
            raise ValueError(
                f"conditions ({len(conditions)}) and n_donors ({len(n_donors)}) "
                "must be the same length.")
        _valid = set(self._all_metadata[self.target_condition].astype(str).unique())
        _unknown = [c for c in map(str, conditions) if c not in _valid]
        if _unknown:
            raise ValueError(
                f"conditions {_unknown} not found in {self.target_condition!r}; "
                f"valid values: {sorted(_valid)}.")
        from m3._engine.simulation import (
            combine_simulated_donors,
            summarise_generated_results,
            synthesize_donors_per_condition,
        )
        # Resolve batch -> (batch_col, target_batch) for the engine.
        batch_col, target_batch = None, -1
        if batch is not None:
            if self.batch_key is None:
                raise ValueError("augment(batch=...) requires batch_key at construction.")
            uniq = self._all_metadata[self.batch_key].astype(str).unique().tolist()
            if str(batch) not in uniq:
                raise ValueError(f"batch {batch!r} not in {self.batch_key} values {uniq}.")
            batch_col, target_batch = self.batch_key, uniq.index(str(batch))
        results = synthesize_donors_per_condition(
            self._generator, self._all_data, self._all_metadata, self._all_b,
            self._all_mask_poe, self.celltype_key, self.donor_key, self.target_condition,
            conditions, n_donors,
            batch_col=batch_col, target_batch=target_batch,
            per_class=-1, tau=tau,
            device=str(self._device), seed=seed)
        summary = summarise_generated_results(results)
        X, meta = combine_simulated_donors(summary)
        # engine emits fixed Liu-style obs column names; map them back to the model's keys
        meta = meta.rename(columns={"cond_group": self.target_condition,
                                    "mergedcelltype": self.celltype_key,
                                    "donor": self.donor_key})
        arr = X.numpy() if hasattr(X, "numpy") else np.asarray(X)
        splits = list(self._generator.encoder.feature_splits)
        present = [m for m in _MODALITY_ORDER if m in self.dataset.modality_names]
        expr, start = {}, 0
        for m, w in zip(present, splits):
            expr[m] = arr[:, start: start + w]
            start += w
        return {"expression": expr, "obs": meta}

    # ---------------------------------------------------------- save / load
    def _fingerprint(self) -> dict:
        """Digest of everything in the Dataset that the model's numbers depend on:
        the counts, the feature names, which batches measure which modality, and
        the obs columns given a role. Keyed by a readable name of each part, so a
        mismatch can say which part differs. Other obs columns are free to change.
        """
        import scipy.sparse as sp

        ds = self.dataset

        def digest(*chunks) -> str:
            h = hashlib.sha256()
            for c in chunks:
                h.update(c)
            return h.hexdigest()

        def text(values) -> bytes:
            # A JSON list keeps the values apart whatever characters they hold, and
            # null keeps a missing label apart from the text "nan" or "None".
            # The type goes in too: 1 and "1" are different labels to the engine.
            return json.dumps([None if (pd.api.types.is_scalar(v) and pd.isna(v))
                               else f"{type(v).__name__}:{v}" for v in values]).encode("ascii")

        batch = ds.obs["batch"].to_numpy()
        fp = {"number of cells": str(ds.n_cells)}
        for m in ds.modality_names:
            mat = sp.csr_matrix(ds.modalities[m], dtype=np.float32, copy=True)
            mat.sum_duplicates()
            mat.eliminate_zeros()
            mat.sort_indices()
            fp[f"counts of '{m}'"] = digest(
                np.asarray(mat.shape, dtype="<i8").tobytes(), mat.indptr.astype("<i8").tobytes(),
                mat.indices.astype("<i8").tobytes(), mat.data.astype("<f4").tobytes())
            fp[f"feature names of '{m}'"] = digest(text(ds.var[m]))
            fp[f"batches measuring '{m}'"] = digest(text(
                bool(ds.present[m][batch == c].any()) for c in ds.batches))
        roles = dict.fromkeys(
            [*self.condition_keys, self.donor_key, self.celltype_key, self.batch_key])
        for col in roles:
            if col is not None:
                fp[f"obs column '{col}'"] = (digest(text(ds.obs[col].tolist()))
                                             if col in ds.obs.columns else "missing")
        return fp

    @staticmethod
    def _fingerprint_diff(a: dict, b: dict) -> list:
        return [k for k in dict.fromkeys([*a, *b]) if a.get(k) != b.get(k)]

    # Outputs recorded in a saved file and recomputed when it is loaded: entry name
    # -> (what a message calls it, tolerance, relative and absolute). On the machine
    # that saved the file they come back identical; on another machine or torch
    # build they differ by float32 rounding (~1e-5), so the tolerances leave two
    # orders of magnitude and still catch a model that computes something else.
    _CHECKS = {
        "probe_inputs": ("model inputs", 1e-5),
        "probe_embedding": ("embedding", 1e-3),
        "probe_embedding_cells": ("embedding", 1e-3),
        "probe_reconstruction_cells": ("reconstruction", 1e-3),
        "probe_reconstruction_features": ("reconstruction", 1e-3),
        "probe_donor_embedding": ("donor embedding", 1e-2),
        "probe_logits": ("donor predictions", 1e-2),
    }

    def _probes(self) -> dict:
        """The outputs save() records and load() recomputes and compares.

        Every cell contributes (its input total, the mean of its embedding and of
        its reconstruction), 256 evenly spaced cells contribute their whole
        embedding, and every donor its corrected vector's mean and its logits.
        Between them they pass through every weight a read-out uses: encoder and
        batch encoder, decoder, corrector and status head.
        """
        n = self._all_data.shape[0]
        sample = np.unique(np.linspace(0, n - 1, num=min(n, 256)).round().astype(np.int64))
        mu = self._forward_mu()[0]
        per_cell, per_feature = [], 0.0
        with torch.no_grad():
            for start in range(0, n, 4096):              # bounded memory on large data
                recon = self._generator.decoder(mu[start:start + 4096])
                per_cell.append(recon.mean(1))
                per_feature = per_feature + recon.double().sum(0)
        out = {"probe_inputs": self._all_data.double().sum(1),
               "probe_embedding": mu[torch.as_tensor(sample)],
               "probe_embedding_cells": mu.mean(1),
               "probe_reconstruction_cells": torch.cat(per_cell),
               "probe_reconstruction_features": per_feature / n}
        if self._corrector is not None:
            donor_corr, logits, _ = self._build_donor_features()
            out["probe_donor_embedding"] = donor_corr.mean(1)
            out["probe_logits"] = logits
        return {k: v.detach().cpu().numpy() for k, v in out.items()}

    def save(self, path) -> str:
        """Write the trained model to ``path``, to be read back with :meth:`load`.

        The file holds the trained weights and the model's settings -- not the
        data. It also records which cells the read-outs cover and a digest of the
        dataset, so that :meth:`load` can rebuild the model from the same Dataset
        and refuse any other. It is a plain zip of arrays (no pickled objects).

        Returns ``path``.
        """
        if self._generator is None:
            raise M3CapabilityError("save requires a trained model; call train() first.")
        path = os.fsdecode(path)
        gen, corr = self._generator, self._corrector

        def plain(value, where):
            if isinstance(value, np.generic):
                value = value.item()
            if not isinstance(value, (str, int, float, bool)):
                raise TypeError(
                    f"cannot save this model: {where} contains a {type(value).__name__} "
                    f"({value!r}), and a model file can only hold text and number labels "
                    "there. Use text labels in that obs column.")
            return value

        held_out = [plain(v, "held_out") for v in self.held_out]
        held_out_samples = [plain(v, "held_out_samples") for v in self.held_out_samples]

        try:
            self.dataset.__post_init__()     # its parts still fit together
            fingerprint = self._fingerprint()
            changed = self._fingerprint_diff(self._trained_on, fingerprint)
            if changed:
                raise ValueError(f"{', '.join(changed)} differ")
        except Exception as e:
            raise ValueError(
                f"the Dataset has changed since this model was trained ({e}), so a saved "
                "copy could not be loaded back; nothing was written. Save before editing "
                "the Dataset.") from e

        # The cells training held out must be the ones the settings ask for, or
        # load() could not tell reference from query.
        asked = self._query_cells
        got = asked[self._all_rows]
        if (got[:self._n_ref].any() or not got[self._n_ref:].all()
                or int(got.sum()) != int(asked.sum())):
            raise ValueError(
                f"this model's held-out set is not the one its settings ask for: held_out / "
                f"held_out_samples name {int(asked.sum())} cells, training held out "
                f"{len(got) - self._n_ref}. Labels that read as numbers (such as '001') "
                "lose their text form in training and then match nothing; rename them and "
                "train again. Nothing was written.")

        # Record which original columns in-model HVG selection kept, so that load()
        # does not depend on re-running the selection.
        present = [m for m in _MODALITY_ORDER if m in self.dataset.modality_names]
        for mod, w in zip(present, gen.encoder.feature_splits):
            full = list(self.dataset.var[mod])
            if self.hvg.get(mod) and w != len(full):
                self._hvg_cols[mod] = [int(i) for i in self._hvg_index(mod, w, full)]

        # load() rebuilds the read-out tensors from the Dataset. Do that here first
        # and compare with the trained model's own: a model that cannot be rebuilt
        # exactly is not written at all.
        data, b, mask, metadata = self._assemble(self._all_rows, self._n_ref)
        roles = list(dict.fromkeys(c for c in (
            *self.condition_keys, self.donor_key, self.celltype_key, self.batch_key, "batch")
            if c is not None))
        if not all(c in self._all_metadata.columns for c in roles) or not (
                metadata[roles].equals(self._all_metadata[roles])):
            raise ValueError(
                "model.cell_metadata has been edited (it is the model's own table, not a "
                f"copy): its columns {roles} no longer match the Dataset, so a saved copy "
                "would not load back to the same model; nothing was written. Work on "
                "model.cell_metadata.copy() instead.")
        if not (torch.equal(data, self._all_data) and torch.equal(b, self._all_b)
                and torch.equal(mask, self._all_mask_poe)):
            raise RuntimeError(
                "m3 could not rebuild this model's inputs from its Dataset, so a saved "
                "copy would not load back to the same model; nothing was written. "
                "Please report this at https://github.com/PYangLab/M3/issues.")

        import m3
        header = {
            "format": _FORMAT,
            "format_version": _FORMAT_VERSION,
            "m3_version": m3.__version__,
            "init": {
                "condition_keys": self.condition_keys,
                "target_condition": self.target_condition,
                "donor_key": self.donor_key,
                "celltype_key": self.celltype_key,
                "batch_key": self.batch_key,
                "held_out": held_out,
                "held_out_samples": held_out_samples,
                "hvg": self.hvg,
                "embedding_dim": self.embedding_dim,
            },
            # Layer widths that do not follow from the settings and the data; the
            # rest are taken from those (and from the stored arrays) on load.
            "generator": {
                "nfeatures": list(gen.encoder.feature_splits),
                "batch_classify_dim": gen.batch_classify_dim,
                "condition_dim": list(gen.condition_dim),
            },
            "donor_predictor": None if corr is None else {"min_cells": self._dp["min_cells"]},
            "n_ref": self._n_ref,
            "hvg_cols": self._hvg_cols,
            "history": self.history,
            "dataset": fingerprint,
        }
        arrays = {"rows": self._all_rows, **self._probes()}
        for k, v in gen.state_dict().items():
            arrays[f"generator/{k}"] = v.detach().cpu().numpy()
        if corr is not None:
            for k, v in corr.state_dict().items():
                arrays[f"corrector/{k}"] = v.detach().cpu().numpy()

        def number(o):                       # numpy scalars in settings / history
            if isinstance(o, np.generic):
                return o.item()
            raise TypeError(f"cannot save a value of type {type(o).__name__} in the model file")

        arrays["header"] = np.array(json.dumps(header, default=number))
        # Write beside the target under a name of its own and rename, so neither an
        # interrupted save nor two saves at once leave a half-written file under
        # the real name.
        tmp = f"{path}.{uuid.uuid4().hex[:12]}.tmp"
        try:
            with open(tmp, "xb") as f:
                np.savez_compressed(f, **arrays)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        return path

    @classmethod
    def load(cls, path, dataset) -> "M3":
        """Read a model written by :meth:`save`.

        ``dataset`` must be the Dataset the model was trained on (the same cells in
        the same order, the same counts, feature names and role columns): the file
        holds the weights, and the read-outs are computed from the data. A
        different dataset, a damaged file, or a file whose weights no longer give
        the outputs recorded at save time is refused with a ``ValueError``.

        The returned model gives the read-outs of the one that was saved
        (``embedding``, ``reconstruct``, ``predict_donors``, ``donor_embedding``,
        ``attribute``, ``generate``, ``augment``). ``generate`` and ``augment`` draw
        random numbers on the model's device, so a GPU and a CPU give different
        draws for the same seed.
        """
        import zipfile

        from m3._dataset import Dataset

        if isinstance(path, Dataset):
            raise TypeError("load(path, dataset): the file path comes first, then the Dataset.")
        path = os.fsdecode(path)
        if not isinstance(dataset, Dataset):
            raise TypeError(
                f"load(path, dataset) needs the m3 Dataset the model was trained on, not "
                f"a {type(dataset).__name__}; build it the way it was built before "
                "training (m3.read_h5ad, m3.from_anndata, m3.read_matrix, ...).")
        if _ROW in dataset.obs.columns:
            raise ValueError(f"obs column '{_ROW}' is reserved by m3; rename it.")

        def unreadable(why):
            return ValueError(f"{path} is not a readable m3 model file ({why}).")

        def reason(e):
            if isinstance(e, _NotAModel):
                return str(e)
            if "pickle" in str(e).lower():
                return "it holds pickled objects, which a model file never does"
            return f"{type(e).__name__}: {e}"

        dataset.__post_init__()              # its parts fit together (ValueError if not)
        if not os.path.isfile(path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"no such file: {path}")
            raise unreadable("it is not a file")
        with open(path, "rb") as f:
            starts_like_a_zip = f.read(4) == b"PK\x03\x04"
        if not starts_like_a_zip or not zipfile.is_zipfile(path):
            raise unreadable("it is not a zip archive")
        try:
            with np.load(path, allow_pickle=False) as z:
                if "header" not in z.files:
                    raise _NotAModel("no header")
                header = json.loads(str(z["header"]))
                if not isinstance(header, dict) or header.get("format") != _FORMAT:
                    raise _NotAModel("no m3 model header")
                version = header.get("format_version")
                if type(version) is not int:
                    raise _NotAModel("its format version is not a whole number")
                extra = [k for k in z.files if k not in ("header", "rows", *cls._CHECKS)
                         and not k.startswith(("generator/", "corrector/"))]
                if extra:
                    raise _NotAModel(f"unexpected entry '{extra[0]}'")
                arrays = {k: z[k] for k in z.files if k != "header"}
        except Exception as e:
            raise unreadable(reason(e)) from e
        if version != _FORMAT_VERSION:
            import m3
            raise ValueError(
                f"{path} uses model file format version {version} (written by m3 "
                f"{header.get('m3_version')}); this m3 ({m3.__version__}) reads format "
                f"version {_FORMAT_VERSION}. Load it with the m3 version that wrote it.")

        try:
            init = dict(header["init"])
            saved_fp = header["dataset"]
            if not isinstance(saved_fp, dict):
                raise _NotAModel("no dataset digest")
        except Exception as e:
            raise unreadable(reason(e)) from e
        wrong_dataset = ("Load it with the same Dataset that was passed to m3.M3(...) when "
                         "the model was trained.")
        try:
            model = cls(dataset, **init)
        except Warning:                      # warnings turned into errors by the caller
            raise
        except ValueError as e:              # the saved settings do not fit this dataset
            raise ValueError(f"{path}: this is not the dataset this model was trained on "
                             f"({e}) {wrong_dataset}") from e
        except Exception as e:
            raise unreadable(reason(e)) from e
        try:
            model._trained_on = model._fingerprint()
        except Exception as e:               # labels the digest cannot even read
            raise ValueError(f"{path}: this is not the dataset this model was trained on "
                             f"({type(e).__name__}: {e}). {wrong_dataset}") from e
        differs = cls._fingerprint_diff(saved_fp, model._trained_on)
        if differs:
            raise ValueError(f"{path}: this is not the dataset this model was trained on "
                             f"({', '.join(differs)} differ). {wrong_dataset}")

        try:
            model._restore(header, arrays)
            recomputed = model._probes()
            if set(recomputed) != {k for k in arrays if k in cls._CHECKS}:
                raise _NotAModel("its recorded outputs are not those of this kind of model")
        except Exception as e:      # missing, misshapen or inconsistent entries
            raise unreadable(reason(e)) from e

        # The weights are in; check that they give the numbers the saved model gave.
        for key, value in recomputed.items():
            what, tol = cls._CHECKS[key]
            saved = arrays[key]
            if saved.shape != value.shape or not np.allclose(
                    value, saved, rtol=tol, atol=tol, equal_nan=True):
                raise ValueError(
                    f"{path}: the loaded model does not reproduce the {what} recorded when "
                    f"it was saved (by m3 {header.get('m3_version')}), so it was not loaded. "
                    "The file may be damaged, or this m3 and the libraries under it compute "
                    "them differently from the ones that saved it.")
        return model

    def _restore(self, header: dict, arrays: dict) -> None:
        """Put a saved model's trained state onto this (freshly constructed) model.

        Shapes come from the settings, the data and the stored arrays themselves;
        whatever the header adds is checked against those before anything is built.
        """
        from m3._engine.model import M3_model
        from m3._engine.patient_prediction import AdvSampleCorrector

        def whole(x):
            if type(x) is not int:
                raise _NotAModel("a recorded size is not a whole number")
            return x

        dev = self._device
        n_ref = whole(header["n_ref"])
        rows = arrays["rows"]
        query = self._query_cells
        if (rows.ndim != 1 or rows.dtype.kind not in "iu" or not 0 < n_ref <= len(rows)
                or rows.min() < 0 or rows.max() >= len(query)
                or len(np.unique(rows)) != len(rows)
                or query[rows[:n_ref]].any() or not query[rows[n_ref:]].all()):
            raise _NotAModel("the recorded cells do not fit the recorded settings")
        rows = rows.astype(np.int64)
        self._hvg_cols = {m: [whole(i) for i in v] for m, v in header["hvg_cols"].items()}
        if not set(self._hvg_cols) <= {m for m in self.hvg if self.hvg[m]}:
            raise _NotAModel("recorded feature selection without an hvg setting")
        self._n_ref, self._all_rows = n_ref, rows
        (self._all_data, self._all_b, self._all_mask_poe,
         self._all_metadata) = self._assemble(rows, n_ref)

        g = header["generator"]
        nfeatures = [whole(v) for v in g["nfeatures"]]
        condition_dim = [whole(v) for v in g["condition_dim"]]
        if (min(nfeatures) < 1 or sum(nfeatures) != self._all_data.shape[1]
                or whole(g["batch_classify_dim"]) != int(self._all_b.max()) + 1
                or len(condition_dim) != len(self.condition_keys)
                or not all(1 <= d <= len(query) for d in condition_dim)):
            raise _NotAModel("the recorded network does not fit the settings and the data")

        def state(prefix):
            return {k[len(prefix):]: torch.from_numpy(v)
                    for k, v in arrays.items() if k.startswith(prefix)}

        gen_state, corr_state = state("generator/"), state("corrector/")
        gen = M3_model(nfeatures, hidden_features=[self.embedding_dim] * len(nfeatures),
                       z_dim=self.embedding_dim,
                       cty_classify_dim=int(gen_state["cty1_classify.net.2.bias"].shape[0]),
                       batch_classify_dim=g["batch_classify_dim"], condition_dim=condition_dim)
        gen.load_state_dict(gen_state)
        self._generator = gen.to(dev).eval()
        self.history = header["history"]

        cfg = header["donor_predictor"]
        if (cfg is None) != (not corr_state):
            raise _NotAModel("donor predictor settings and weights do not go together")
        self.capabilities = {"embedding": True, "reconstruct": True,
                             "predict_donors": cfg is not None}
        if cfg is not None:
            self._donor_setup(cfg["min_cells"])
            dp = self._dp
            dp["hidden"] = int(corr_state["transform.0.0.bias"].shape[0])
            corr = AdvSampleCorrector(
                dim=dp["n_celltypes"] * self._emb_dim, n_con=len(dp["condition_vocab"]),
                n_batch=dp["n_site"], hidden=dp["hidden"], p=0.1, bn=False)
            corr.load_state_dict(corr_state)
            self._corrector = corr.to(dev).eval()

    @property
    def cell_metadata(self):
        """Row-aligned metadata for embedding/reconstruction rows (ref+query order)."""
        return self._all_metadata


_HOUSEKEEPING_RE = r"^MT-|^MTMR|^MTND|^MT[A-Z]|^RPL|^RPS|^RP[0-9]"


class Attribution:
    """Integrated-gradients attribution result with ranked tables.

    Three views of gene importance are available:

    * ``self.genes`` -- raw engine output: ``mean(|IG|)`` across ALL cells. Quick
      and unfiltered; use this when you don't have a target/reference condition.
    * ``self.gene_celltype_matrix`` -- signed per-(cell-type, gene) attribution,
      the substrate for per-celltype-balanced ranking.
    * ``self.top_genes(...)`` -- the publication recipe:
      drop cell types where either condition has < ``min_cells_per_condition``
      cells, score each gene as ``mean(|gene_celltype_matrix|)`` across kept
      cell types, exclude housekeeping/ribosomal genes by name, optionally mask
      modalities, and return the top-N ranked DataFrame.
    """

    def __init__(self, res: dict, feature_names: list, target_label: str | None = None,
                 modality_of: list | None = None,
                 cell_metadata: pd.DataFrame | None = None,
                 celltype_key: str | None = None,
                 donor_key: str | None = None,
                 target_condition: str | None = None,
                 reference_labels: list | None = None):
        self.target_label = target_label
        self._feature_names = list(feature_names)
        self._modality_of = list(modality_of) if modality_of is not None else None
        self._cell_metadata = cell_metadata
        self._celltype_key = celltype_key
        self._donor_key = donor_key
        self._target_condition = target_condition
        self._reference_labels = list(reference_labels) if reference_labels else None

        def _np(x):
            return x.numpy() if hasattr(x, "numpy") else np.asarray(x)

        gi = _np(res["gene_importance"]).ravel()
        self.genes = (pd.DataFrame({"feature": self._feature_names[: len(gi)], "importance": gi})
                      .sort_values("importance", ascending=False).reset_index(drop=True))
        ci = _np(res["celltype_importance"]).ravel()
        self.celltypes = (pd.DataFrame({"celltype": list(res["celltype_names"]), "importance": ci})
                          .sort_values("importance", ascending=False).reset_index(drop=True))
        if "donor_attribution" in res:
            da = _np(res["donor_attribution"]).ravel()
            names = list(res.get("donor_names", range(len(da))))
            self.donors = (pd.DataFrame({(self._donor_key or "donor"): names[: len(da)], "attribution": da})
                           .sort_values("attribution", ascending=False).reset_index(drop=True))
        else:
            self.donors = None
        self.cells = _np(res["cell_importance"]).ravel()
        self.attribution = _np(res["attribution"])               # cell x gene
        self.gene_celltype_matrix = _np(res["gene_celltype_matrix"])   # celltype x gene
        self._res = res

    def top_celltypes(self, *, min_cells_per_condition: int = 200) -> pd.DataFrame:
        """Cell-type importance ranking, filtered to types with enough cells in
        both conditions.

        min_cells_per_condition: drop cell types where the reference condition
            or the target condition has fewer than this many cells
            (default 200, matches the Liu publication recipe). Set to 0 to
            return the raw :attr:`celltypes` table unfiltered.

        Note: this does NOT exclude QC labels like 'Unk' / 'dblt' / 'dim' --
        those are dataset-specific QC categories that should be cleaned out at
        the data-processing stage before calling ``m3.M3(...)``.
        """
        if min_cells_per_condition <= 0:
            return self.celltypes.copy()
        if self._cell_metadata is None or self._celltype_key is None or \
           self._target_condition is None or self._reference_labels is None:
            raise ValueError(
                "top_celltypes with min_cells_per_condition>0 needs cell_metadata, "
                "celltype_key, target_condition, and reference_labels -- "
                "obtain the Attribution via M3.attribute(...).")
        meta = self._cell_metadata
        ref_set = set(map(str, self._reference_labels))
        is_ref = meta[self._target_condition].astype(str).isin(ref_set).to_numpy()
        cty_str = meta[self._celltype_key].astype(str)

        keep = []
        for ct in self.celltypes["celltype"].astype(str).to_list():
            in_ct = (cty_str == ct).to_numpy()
            keep.append(int((in_ct & is_ref).sum()) >= min_cells_per_condition
                        and int((in_ct & ~is_ref).sum()) >= min_cells_per_condition)
        return self.celltypes.loc[keep].reset_index(drop=True)

    def top_genes(self, n: int = 100, *,
                  min_cells_per_condition: int = 200,
                  exclude_regex: str | None = _HOUSEKEEPING_RE,
                  modality: str | None = None) -> pd.DataFrame:
        """Per-celltype-balanced top genes (publication recipe).

        n: how many genes to keep (default 100).
        min_cells_per_condition: drop cell types where either condition has fewer
            than this many cells (e.g. 200 in the Liu pipeline). Set to 0 to skip.
        exclude_regex: regex matched against feature names to drop housekeeping /
            ribosomal genes. Default matches ^MT-, ^MTMR, ^MTND, ^MT[A-Z], ^RPL,
            ^RPS, ^RP[0-9]. Pass ``None`` to skip.
        modality: 'rna' / 'adt' / 'atac' to restrict ranking to one modality
            (requires the Attribution to know per-feature modality; ``None`` =
            no modality filter).
        """
        import re

        gcm = self.gene_celltype_matrix                   # [n_celltypes, n_genes]
        celltype_names = list(self._res["celltype_names"])
        keep_ct_mask = np.ones(len(celltype_names), dtype=bool)

        # Filter cell types by min cells per condition (HC vs target)
        if min_cells_per_condition > 0:
            if self._cell_metadata is None or self._celltype_key is None or \
               self._target_condition is None or self._reference_labels is None:
                raise ValueError(
                    "top_genes with min_cells_per_condition>0 needs cell_metadata, "
                    "celltype_key, target_condition, and reference_labels -- "
                    "obtain the Attribution via M3.attribute(...).")
            meta = self._cell_metadata
            ref_set = set(map(str, self._reference_labels))
            cond_str = meta[self._target_condition].astype(str)
            is_ref = cond_str.isin(ref_set)
            cty_str = meta[self._celltype_key].astype(str)
            for i, ct in enumerate(celltype_names):
                in_ct = (cty_str == str(ct))
                n_ref = int((in_ct & is_ref).sum())
                n_tgt = int((in_ct & ~is_ref).sum())
                keep_ct_mask[i] = (n_ref >= min_cells_per_condition
                                   and n_tgt >= min_cells_per_condition)

        if not keep_ct_mask.any():
            raise ValueError(
                f"no cell types meet min_cells_per_condition={min_cells_per_condition}.")

        # Per-celltype-balanced score: mean(|gene_celltype_matrix|) over kept cell types
        score = np.abs(gcm[keep_ct_mask]).mean(axis=0)

        # Mask housekeeping by regex
        if exclude_regex:
            pat = re.compile(exclude_regex)
            for i, name in enumerate(self._feature_names):
                if pat.search(name):
                    score[i] = -np.inf

        # Modality mask
        if modality is not None:
            if self._modality_of is None:
                raise ValueError(
                    "modality filter requested but per-feature modality is unknown.")
            for i, m in enumerate(self._modality_of):
                if m != modality:
                    score[i] = -np.inf

        # Build ranking
        kept = score > -np.inf
        if not kept.any():
            raise ValueError("no genes survived the filters.")
        order = np.argsort(-score)
        order = [i for i in order if score[i] > -np.inf][: n]

        return pd.DataFrame({
            "feature": [self._feature_names[i] for i in order],
            "modality": [self._modality_of[i] if self._modality_of else "?"
                         for i in order],
            "score": score[order],
            "n_celltypes_used": int(keep_ct_mask.sum()),
        }).reset_index(drop=True)

    def __repr__(self):
        nd = None if self.donors is None else len(self.donors)
        return (f"Attribution(target={self.target_label!r}, genes={len(self.genes)}, "
                f"celltypes={len(self.celltypes)}, donors={nd})")
