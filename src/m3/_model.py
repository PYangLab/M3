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

import shutil
import tempfile
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from m3 import _bridge

_MODALITY_ORDER = ("rna", "adt", "atac")


class M3CapabilityError(RuntimeError):
    """Raised when a readout needs a capability the model has not been trained for."""


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
        for col in condition_keys:
            if col not in dataset.obs.columns:
                raise ValueError(f"condition key '{col}' not in dataset.obs.")
        for nm, col in (("celltype_key", celltype_key), ("donor_key", donor_key),
                        ("batch_key", batch_key)):
            if col is not None and col not in dataset.obs.columns:
                raise ValueError(f"{nm} '{col}' not in dataset.obs.")

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
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._all_data = self._all_b = self._all_mask_poe = None
        self._all_metadata = None
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
        which the upstream engine otherwise leaves unseeded — without it a run
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

        ds = self.dataset
        if self._query_cells.any() and any(
                ds.obs.loc[self._query_cells, k].isna().any() for k in self.condition_keys):
            # Held-out cells' condition labels are masked in training anyway; fill
            # any NaN with a reference placeholder so the engine sees a valid
            # category (no requirement to supply query labels).
            import dataclasses
            patched = ds.obs.copy()
            for k in self.condition_keys:
                ph = self.contract["reference_vocab"][k][0]
                col = patched[k].astype(object)
                col[self._query_cells & patched[k].isna().to_numpy()] = ph
                patched[k] = col
            ds = dataclasses.replace(ds, obs=patched)
        marshalled = _bridge.marshal(ds)
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
        # drift); seed here — mirroring the R interface — so the run is reproducible
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

        self._all_data = self._all_data.to(self._device)
        self._all_b = self._all_b.to(self._device).long()
        self._all_mask_poe = self._all_mask_poe.to(self._device)
        shutil.rmtree(marshalled["tmpdir"], ignore_errors=True)

        self.capabilities["embedding"] = True
        self.capabilities["reconstruct"] = True

        if want_donor:
            self._fit_donor_predictor(donor_predictor or {})
            self.capabilities["predict_donors"] = True
        return self

    # ---------------------------------------------- Stage-2 donor predictor
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
        # engine's full-set ordering — true for the shipped setup (string disease labels,
        # both classes present in the reference). For numeric-coded conditions or a
        # query-exclusive category the code spaces can diverge for the auxiliary cond_loss.
        ref_meta = meta.iloc[:n_ref]
        ref_cond_cat = pd.Categorical(ref_meta[self.target_condition].astype(str))
        condition_vocab = [str(c) for c in ref_cond_cat.categories]
        n_cond_classes = len(condition_vocab)
        code_of = {v: i for i, v in enumerate(condition_vocab)}
        cond_codes = torch.tensor(
            [code_of.get(s, -1) for s in meta[self.target_condition].astype(str)],
            dtype=torch.long).to(dev)

        site_cat = pd.Categorical(meta[self.site_col].astype(str))
        n_site = len(site_cat.categories)
        site_codes_cell = torch.tensor(site_cat.codes.astype("int64").copy()).to(dev)
        is_ref = torch.zeros(n_total, dtype=torch.bool, device=dev)
        is_ref[:n_ref] = True

        unique_donors0 = torch.unique(donor_codes)
        donor_status, donor_site, donor_is_ref = {}, {}, {}
        for dc in unique_donors0.tolist():
            m = (donor_codes == dc)
            donor_status[dc] = int(cond_codes[m][0])
            donor_site[dc] = int(site_codes_cell[m][0])
            donor_is_ref[dc] = bool(is_ref[m][0])

        def agg_diff(z):
            uds = torch.unique(donor_codes)
            rows = []
            for dc in uds:
                parts = [z[(donor_codes == dc) & (celltype_codes == k)].mean(0)
                         if int(((donor_codes == dc) & (celltype_codes == k)).sum()) >= min_cells
                         else z.new_zeros(emb_dim)
                         for k in range(n_celltypes)]
                rows.append(torch.cat(parts))
            return torch.stack(rows), uds

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
            hist.append({"epoch": epoch, "cell_loss": float(cell_loss),
                         "status_loss": float(status_loss), "adv_w": float(adv_w)})

        self._corrector = corrector
        self.history = hist
        self._dp = {
            "donor_cat": donor_cat, "condition_vocab": condition_vocab,
            "donor_codes": donor_codes, "celltype_codes": celltype_codes,
            "donor_is_ref": donor_is_ref, "donor_status": donor_status,
            "n_celltypes": n_celltypes, "agg_diff": agg_diff,
        }

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
            donor_feat, uds = self._dp["agg_diff"](z)
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
        rows = {"donor": names, "is_reference": ~is_query,
                "predicted_label": [vocab[i] for i in pred]}
        for i, v in enumerate(vocab):
            rows[f"prob_{v}"] = prob[:, i]
        df = pd.DataFrame(rows)
        if not include_reference:
            df = df[~df["is_reference"]].reset_index(drop=True)
        return df

    def donor_embedding(self) -> pd.DataFrame:
        """Patient/donor-level embedding — the corrector-corrected donor vector used for
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
        df = pd.DataFrame(emb, index=pd.Index(names, name="donor"),
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
            generation are restricted to that batch — useful for reproducing
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

    @property
    def cell_metadata(self):
        """Row-aligned metadata for embedding/reconstruction rows (ref+query order)."""
        return self._all_metadata


_HOUSEKEEPING_RE = r"^MT-|^MTMR|^MTND|^MT[A-Z]|^RPL|^RPS|^RP[0-9]"


class Attribution:
    """Integrated-gradients attribution result with ranked tables.

    Three views of gene importance are available:

    * ``self.genes`` — raw engine output: ``mean(|IG|)`` across ALL cells. Quick
      and unfiltered; use this when you don't have a target/reference condition.
    * ``self.gene_celltype_matrix`` — signed per-(cell-type, gene) attribution,
      the substrate for per-celltype-balanced ranking.
    * ``self.top_genes(...)`` — the publication recipe:
      drop cell types where either condition has < ``min_cells_per_condition``
      cells, score each gene as ``mean(|gene_celltype_matrix|)`` across kept
      cell types, exclude housekeeping/ribosomal genes by name, optionally mask
      modalities, and return the top-N ranked DataFrame.
    """

    def __init__(self, res: dict, feature_names: list, target_label: str | None = None,
                 modality_of: list | None = None,
                 cell_metadata: pd.DataFrame | None = None,
                 celltype_key: str | None = None,
                 target_condition: str | None = None,
                 reference_labels: list | None = None):
        self.target_label = target_label
        self._feature_names = list(feature_names)
        self._modality_of = list(modality_of) if modality_of is not None else None
        self._cell_metadata = cell_metadata
        self._celltype_key = celltype_key
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
            self.donors = (pd.DataFrame({"donor": names[: len(da)], "attribution": da})
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

        Note: this does NOT exclude QC labels like 'Unk' / 'dblt' / 'dim' —
        those are dataset-specific QC categories that should be cleaned out at
        the data-processing stage before calling ``m3.M3(...)``.
        """
        if min_cells_per_condition <= 0:
            return self.celltypes.copy()
        if self._cell_metadata is None or self._celltype_key is None or \
           self._target_condition is None or self._reference_labels is None:
            raise ValueError(
                "top_celltypes with min_cells_per_condition>0 needs cell_metadata, "
                "celltype_key, target_condition, and reference_labels — "
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
                    "celltype_key, target_condition, and reference_labels — "
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
