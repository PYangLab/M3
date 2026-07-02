---
hide:
  - navigation
  - toc
---

<div class="lp" markdown>

<section class="lp-hero" markdown>

<div class="lp-hero-text" markdown>

# M3

<p class="lp-sub">
  A deep generative framework that learns factorised, condition-aware embeddings
  for multimodal, multi-condition, multi-sample single-cell data, separating
  biological signal from condition and batch effects so integration,
  patient-level inference, and attribution stay internally consistent.
</p>

```bash
pip install "git+https://github.com/PYangLab/M3.git"
```

<a href="quickstart/" class="md-button md-button--primary lp-cta">Get started</a>
<a href="overview/" class="md-button lp-cta">How it works</a>

</div>

<div class="lp-hero-card">
<div class="taskflow" data-taskflow="m3">
<div class="tf-main">
<div class="tf-head">
<span class="tf-num">1</span>
<span class="tf-title">Factorised dimension reduction</span>
</div>
<div class="tf-stage">
<svg viewBox="0 0 460 210" xmlns="http://www.w3.org/2000/svg" font-family="Inter, sans-serif" aria-hidden="true"><g class="tf-scenes"></g></svg>
</div>
<div class="tf-dots"><span class="is-active"></span></div>
</div>
</div>
</div>

</section>

<section class="lp-section" markdown>
<div class="lp-grid-4" markdown>

<div class="lp-card lp-card-link">
<a class="lp-card-cover" href="notebooks/py_01_representation_learning/">
<span class="lp-card-kicker">Integrate</span>
<span class="lp-card-title">Representation learning</span>
<span class="lp-card-desc">Learn factorised, condition-aware cell embeddings that integrate across batches, conditions, and modalities while preserving condition signal.</span>
</a>
</div>

<div class="lp-card lp-card-link">
<a class="lp-card-cover" href="notebooks/py_02_patient_prediction/">
<span class="lp-card-kicker">Predict</span>
<span class="lp-card-title">Patient-level inference</span>
<span class="lp-card-desc">Aggregate cell embeddings into pseudo-bulk profiles to predict a sample's condition, such as disease status, on unseen query data.</span>
</a>
</div>

<div class="lp-card lp-card-link">
<a class="lp-card-cover" href="notebooks/py_03_attribution/">
<span class="lp-card-kicker">Interpret</span>
<span class="lp-card-title">Feature attribution</span>
<span class="lp-card-desc">Attribute a condition back to genes, individual cells, and cell types for multi-resolution interpretation.</span>
</a>
</div>

<div class="lp-card lp-card-link">
<a class="lp-card-cover" href="notebooks/py_04_augmentation/">
<span class="lp-card-kicker">Generate</span>
<span class="lp-card-title">Data augmentation</span>
<span class="lp-card-desc">Synthesise new donors and cells to augment small cohorts directly from the learned representation, with batch-stratified control.</span>
</a>
</div>

</div>
</section>

<section class="lp-section lp-section--band" markdown>
<div class="lp-feature-row">
<div class="lp-feat">
<span class="lp-feat-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3 3 7.5 12 12 21 7.5 12 3Z"/><path d="M3 12 12 16.5 21 12"/><path d="M3 16.5 12 21 21 16.5"/></svg></span>
<span class="lp-feat-title">Factorised, condition-aware</span>
<span class="lp-feat-desc">M3 learns embeddings that separate three sources of variation explicitly, biological signal, condition effect, and batch effect, so batch correction does not overcorrect the signal you actually care about.</span>
</div>
<div class="lp-feat">
<span class="lp-feat-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3 3 7.5 12 12 21 7.5 12 3Z"/><path d="M3 12 12 16.5 21 12"/><circle cx="18" cy="6" r="2.5"/></svg></span>
<span class="lp-feat-title">Multimodal and mosaic</span>
<span class="lp-feat-desc">A product-of-experts fuses modality-specific encoders into one shared biological embedding per cell, handling both fully observed and mosaic designs where some samples are missing a modality.</span>
</div>
<div class="lp-feat">
<span class="lp-feat-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="10.5" cy="10.5" r="6"/><path d="M19.5 19.5l-4.6-4.6"/><path d="M8.5 12v-1.5M10.5 12V8.5M12.5 12v-2.5"/></svg></span>
<span class="lp-feat-title">Interpretable across tasks</span>
<span class="lp-feat-desc">Every result decodes from one shared, condition-aware representation, keeping integration, patient-level inference, generation, and gene-, cell-, and cell-type-level attribution internally consistent.</span>
</div>
</div>
</section>

</div>
