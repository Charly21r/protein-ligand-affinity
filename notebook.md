# Lab notebook

One dated entry per working session. What I tried, what happened, conclussions. 

---

## 2026-09-15 — Set up repo and scaffolding

**Tried.** Set up the repository: directory layout (`src/lba/{data,models,negatives,eval}`, `scripts/`, `configs/`, `results/`, `notebooks/`), `pyproject.toml`, `scripts/setup_env.sh`,
`scripts/check_env.py`. Wrote README.md with the hypothesis, the three hard parts
(negatives, family labels, tuning lambda), etc. Drew the architecture
figure (`assets/adversarial_affinity_training_architecture.png`).

---

## 2026-09-16 — Environment and first data load

**Tried.** Installed the environment and downloaded ATOM3D LBA, 30% sequence-identity split
(`scripts/download_data.py`, ~1 GB from Zenodo). Wrote `scripts/smoke_test_lba.py`.

**What happened.** Hit a silent data-corruption bug: on pandas 3.0 every `atoms_*` field loads
as a raw dict instead of a DataFrame. Cause: atom3d only rebuilds the serialized frames when a
type tag stored in the LMDB matches `str(pd.DataFrame)`, and pandas 3.0 changed that string.
No error is raised — the data just comes back wrong. Pinned `pandas<3.0` and declared `pyarrow`.

Smoke test passes: splits are 3507 / 466 / 490 as expected. Labels (`scores.neglog_aff`,
i.e. pKd/pKi) span 2.00–11.92, mean 6.52, train SD 2.00. Pocket sizes median ~367 atoms
(including hydrogens), ligands median 26 atoms, full proteins median 4259 atoms.

**Concluded.** Environment and data are ready. The pandas pin is load-bearing and non-obvious, anyone reproducing this on a fresh environment will hit it.

---

## 2026-09-18 — Reviewing the ligand-only implementation

**Tried.** Implemented the ligand-only code (data loader, featurizers, metrics, training script).

**Concluded.** The implementation is sound in its core logic. 

---

## 2026-09-19 — Ligand-only implementation landed

**Tried.** Committed `src/lba/data/lba.py` (LMDB → SMILES + affinity, parquet-cached) and
`src/lba/models/ligand_only.py` (descriptors, Morgan fingerprints, Tanimoto nearest-neighbour).

**Design note.** Three featurizations chosen deliberately as a ladder of how much each can
memorize: descriptors *describe* but cannot identify a molecule; Morgan fingerprints encode
substructure identity and so act as a fuzzy lookup key; the nearest-neighbour probe is pure
retrieval with no learning at all. The gaps between the three are the evidence, not the
individual numbers.

---

## 2026-09-22 — First results, and the descriptor finding

**Tried.** Committed metrics and the training script, ran the full ligand-only baseline on the
30% split.

**Results (test split, RF, 3 seeds).**

| Model | Featurization | r | RMSE | SD |
|-------|----------------|------|--------|--------|
| rf | descriptors_full | +0.516 | 1.466 | 0.004 |
| ridge | descriptors_full | +0.503 | 1.525 | — |
| rf | descriptors | +0.467 | 1.529 | 0.002 |
| rf | morgan | +0.454 | 1.531 | 0.002 |
| ridge | descriptors | +0.420 | 1.555 | — |
| ridge | morgan | +0.364 | 1.780 | — |
| nn_tanimoto_k1 | morgan | +0.339 | 2.067 | — |
| predict_mean | — | +0.000 | 1.705 | — |

**Finding: the curated descriptor set was costing performance.** The code used 20
hand-picked descriptors (Lipinski + Veber + classical topological indices), justified in a
comment on the grounds that the full RDKit set is "degenerate or NaN-prone." Tested it: of
217 descriptors only 9 ever go non-finite (0.65% of cells, all on relaxed-parsed ligands) and
10 are constant. Training on all 217 instead:

| Descriptor Set | Val r | Test r |
|---|---|---|
| curated 20 | 0.516 ± 0.004 | 0.467 ± 0.001 |
| all 217 | 0.564 ± 0.002 | 0.516 ± 0.003 |

Gap of ~0.05 on both splits, far outside seed noise. Reason: the curated 20 are massively
collinear — 20 pairs above |r| = 0.9, and six of them (MolWt, HeavyAtomMolWt, HeavyAtomCount,
LabuteASA, MolMR, BertzCT) are all measuring *size*. The full set adds genuinely different
information (EState indices, Chi connectivity, PEOE_VSA partial-charge surface areas). Kept
both: the curated set for the interpretable story, the full set as the honest ceiling.

**Added.** Exact-SMILES-duplicate logging to the retrieval probe. **13.9% of test ligands and
12.0% of val ligands appear verbatim in training** — the 30% split separates proteins, not
ligands, so ligand leakage survives it.

**Concluded.** Ligand-only headline is **r = 0.516, RMSE 1.466** (this is not near Volkov's
0.749, and it should not be: that figure is a different dataset on a more permissive split).

---

## 2026-09-24 — Interpreting the baseline

**Tried.** Three analyses to work out *what* the ligand-only signal actually is, since the
headline number alone cannot distinguish memorization from generalization.

**1. Is it memorization?** Stratified test performance by each test ligand's maximum Tanimoto
similarity to any training ligand (rf, all 217 descriptors):

| Similarity Stratum | n | r | RMSE |
|---|---:|---:|---:|
| 0.0–0.4 (novel) | 278 | +0.465 | 1.484 |
| 0.4–0.7 | 116 | +0.553 | 1.477 |
| 0.7–0.9 | 23 | +0.612 | 1.242 |
| 0.9–1.0 (near-dup) | 73 | +0.535 | 1.441 |
| exact SMILES duplicate | 68 | +0.565 | — |
| everything else | 422 | +0.512 | — |

Milder gradient than expected. Performance degrades gracefully on novel ligands rather than
collapsing, so the model is **not** primarily a lookup table. Good news for the pipeline.

**2. Then what is it?** Mostly size:

| Model | Test r |
|---|---:|
| HeavyAtomCount alone, no model | +0.435 |
| rf on all 217 descriptors | +0.514 |

Counting the ligand's atoms gets 85% of the correlation. This is the Kuntz (1999) maximal-affinity
relationship, and it is the reason ligand efficiency exists as a metric — medicinal chemistry
decided decades ago that raw affinity is too confounded by size to compare molecules on. A
benchmark scored by Pearson r hands ~0.44 to anyone who can count atoms.

**3. How much affinity does the ligand genuinely fail to explain?** Measured the spread of
affinities for the *same* molecule across different proteins. 453 distinct ligands appear in
more than one complex (1432 complexes):

| Metric | Value |
|---|---|
| Median within-ligand label SD | 0.72 pKd units |
| Median within-ligand range | 1.13 |
| Max within-ligand range | 6.43 |

The extreme case is an amprenavir-like inhibitor in 23 complexes, spanning pKd 5.27 to 11.70 —
the identical molecule, a ~2.5-million-fold difference in binding, determined entirely by which
protein it sits in. A perfect ligand-only oracle (predicting each ligand's own mean) would
still hit **RMSE 0.958** on repeated ligands. That is the irreducible error of ignoring the
protein, and the headroom an interaction-aware model has to claim.

**4. Statistical resolution.** With n = 490, the 95% CI on a single Pearson r is about
**±0.065** (r = 0.516 → [0.448, 0.578]). Seed variance (±0.004) is an order of magnitude
smaller and does *not* address this — it measures optimizer noise, not test-set sampling noise.
Comparisons on the same test set are paired, which helps a lot: simulated paired bootstrap
(rho = 0.9 between model predictions) detects a true delta-r of 0.05 about 94% of the time,
0.03 about 54%, 0.02 about 34%.

**Concluded.**
- Ligand-only target to beat: **r = 0.516 / RMSE 1.466**, by **delta-r >= 0.05**
  or it isn't resolvable on this test set.
- Published 3D models sit near r ~= 0.55 on this split, i.e. ~+0.04 over ligand-only — right at the edge of what 490 test complexes can resolve. Expect the "full model barely beats ligand-only" outcome.
- **Build paired-bootstrap CIs from the start.** With six configurations the temptation will be to sort by point estimate and narrate the ordering; most of that ordering will not be real.
- Best follow-up once the GVP-GNN exists: re-run analysis 1 on it. If the full model's advantage shows up specifically on the *novel-ligand* stratum, that is positive evidence it is using the pocket. Flat, or concentrated on near-duplicates, means it isn't.
- Report RMSE alongside r throughout. The retrieval probe has r = +0.339 but R2 = −0.506 — it ranks better than chance while being worse than a constant prediction, because it is badly miscalibrated (prediction SD 1.85 against a label SD of 1.68). Pearson r alone would have hidden that.

**Caveat to carry forward.** The size effect may be partly a curation artifact rather than pure
thermodynamics: weak binders in PDBbind are often small fragments from fragment screens, strong
binders are optimized drugs, which are larger. Both make it a shortcut, but I cannot fully
separate the two. Worth a sentence in the report.

---

## 2026-09-24 (cont.) — Pocket-only baseline, and a benchmark problem

**Tried.** Built the pocket-only baseline: `src/lba/models/pocket_only.py`
and `scripts/train_pocket_baseline.py`. Features from `atoms_pocket` only, ligand never seen:
20 amino-acid fractions, an "other" bucket, 11 ion presence flags, plus size/shape
(n_residues, n_heavy_atoms, radius of gyration, principal-axis extents). Hydrogens dropped
(~52% of pocket atoms, computationally added, not experimental). Shape features are
rotation-invariant by construction — verified a random rotation+translation leaves them
identical to 1e-8, since per-axis bounding boxes would have encoded the crystal frame.

**Anticipated confound, confirmed.** ATOM3D defines the pocket by proximity to the ligand, so
pocket size is a proxy for ligand size: corr(pocket residues, ligand heavy atoms) = +0.695
(full training split; an earlier spot-check on 600 complexes gave +0.676).
Ran two feature sets to separate it.

**Results (test, RF, 3 seeds).**

| Model | Features | r | RMSE | Notes |
|---|---|---:|---:|---|
| rf | composition_size | +0.595 | 1.359 | includes the confound |
| ridge | composition_size | +0.535 | 1.427 | |
| rf | composition | +0.475 | 1.484 | size-free, the honest number |
| ridge | composition | +0.147 | 1.695 | |
| predict_mean | — | +0.000 | 1.705 | |

Size confound is worth **delta r = +0.121**.

Note ridge 0.147 vs RF 0.475 on the same size-free features: the composition-to-affinity
relationship is strongly nonlinear, and a linear model gets almost nothing from it.

**The finding that matters.** Fitted single features directly:

| Feature | Test r | RMSE |
|---|---:|---:|
| n_residues | +0.576 | 1.387 |
| n_heavy_atoms | +0.556 | 1.405 |
| radius_of_gyration | +0.497 | 1.474 |

**Counting the residues in the pocket — one integer, one linear fit — scores r = 0.576 /
RMSE 1.387.** That beats my entire 217-descriptor ligand-only model (0.516 / 1.466) and is
around the level published 3D deep models reach on this split (~0.55).

**Paired bootstrap against ligand-only** (3000 resamples, same 490 test complexes, verified
identical pdb_id ordering):

| Comparison | Delta r | 95% CI | Significance |
|---|---:|---|---|
| pocket+size − ligand-only | +0.080 | [+0.010, +0.150] | significant |
| pocket-comp − ligand-only | −0.040 | [−0.105, +0.027] | not resolvable |

**Concluded.**
- Volkov replicated on both sides: ligand-only 0.516 and size-free pocket-only 0.477 are
  statistically indistinguishable. Each half of the complex, alone, predicts affinity about
  as well as the other. Neither sees an interaction.
- The "pocket beats ligand" result is **driven by the confound**, not by protein chemistry.
  Do not claim protein identity beats ligand identity — the honest comparison is the size-free
  one, and it is a tie.
- **The benchmark critique is the real result.** A single integer that requires no chemistry,
  no learning and no ligand structure gets r = 0.576. And critically: `n_residues` is only
  computable *because the pocket was cut around a known ligand pose*. In the real prospective
  task — given a protein, predict affinity for a new compound — that feature cannot exist.
  So the strongest single predictor on this benchmark is an artifact of how the benchmark was
  built. This goes in the report.
- Practical consequence: any full model must be checked against `n_residues`
  (r = 0.576), not just against predict-the-mean. A GVP-GNN scoring 0.58 would have learned
  nothing a residue count does not already give.

**TODO.** Pocket-only inherits the same caveat as ligand size: is this thermodynamics
(bigger pocket, bigger ligand, more contact, tighter binding) or PDBbind curation (fragments
vs optimised drugs)? Still cannot separate them.

---

## 2026-09-29 — Making the analysis reproducible

**Built.**

- `src/lba/eval/stats.py` — `pearson_ci` (Fisher-z interval), `paired_bootstrap_delta`
  (paired CI on the difference between two models, vectorized over resamples, supports
  pearson/spearman/rmse/mae), `calibration_diagnostics` (bias, prediction SD, calibration
  slope, r-squared vs R2 gap).
- `src/lba/eval/probes.py` — `stratify_by_similarity`, `within_ligand_variance`,
  `single_feature_scan`, `single_feature_baseline`, `collinear_pairs`,
  `smiles_quality_report`.
- `scripts/analyze_baselines.py` — one command, writes 12 `results/tables/analysis_*.csv`.


**Everything reproduced.** Affinity skew −0.8317, zero stereo, 546 relaxed parses, 453
repeated ligands over 1432 complexes, within-ligand SD 0.716, oracle RMSE 0.958, MolMR 0.503, 20 collinear pairs, ligand r = 0.516 CI [0.448, 0.578], calibration slope 0.771, paired deltas +0.080 and −0.040. Updated the stale figures in the entries above to match the generated tables.

**Gap the cross-check exposed.** The feature scan runs on *train* and gives `n_residues`
r = 0.469, but the 0.576 quoted on 2026-09-24 is a linear model *fitted on train and scored on test* — a different quantity, and the one that matters. Added `single_feature_baseline` so it is regenerated directly:

| Feature | Source | Pearson r | RMSE | R² |
|---|---|---:|---:|---:|
| n_residues | pocket | 0.576 | 1.387 | 0.322 |
| n_heavy_atoms | pocket | 0.556 | 1.405 | 0.304 |
| radius_of_gyration | pocket | 0.497 | 1.474 | 0.234 |
| LabuteASA | ligand | 0.441 | 1.512 | 0.194 |
| HeavyAtomCount | ligand | 0.435 | 1.517 | 0.189 |

`n_residues` alone reaches **R2 = 0.322** against 0.243 for the full 217-descriptor ligand model. One integer explains a third more variance than the entire ligand pipeline.

**Confound is larger than first measured.** On the full training split rather than a
600-complex spot-check:

| Feature | r with ligand heavy atoms | r with pKd |
|---|---:|---:|
| n_residues | 0.695 | 0.469 |
| n_heavy_atoms | 0.663 | 0.447 |
| radius_of_gyration | 0.745 | 0.403 |

Radius of gyration is the worst at 0.745 and I had not checked it before.

**New finding — the two baselines fail in opposite directions.** Stratifying by ligand
novelty, which I had only run on the ligand model before:

| Similarity Stratum | Ligand-only | Pocket+size |
|---|---:|---:|
| 0.0–0.4 (novel) | +0.465 | +0.606 |
| 0.4–0.7 | +0.553 | +0.557 |
| 0.7–0.9 | +0.612 | +0.660 |
| 0.9–1.0 (near-dup) | +0.535 | +0.538 |

The ligand model is *worst* on novel ligands, as expected. The pocket model is **best** on
them (0.606), because its features do not depend on ligand similarity at all — the novel
stratum is simply where the pocket-size signal happens to be cleanest. Consequence: the
pocket baseline's advantage over ligand-only is concentrated exactly where a real
interaction model would need to prove itself. The interaction-aware baseline has to beat 0.606 on the novel stratum, not 0.516 overall.

**Calibration.** `calibration_diagnostics` on all three baselines:

| Model | Slope | r² | R² | Gap |
|---|---:|---:|---:|---:|
| ligand_descriptors_full | 0.771 | 0.267 | 0.243 | 0.024 |
| pocket_composition | 1.063 | 0.227 | 0.225 | 0.002 |
| pocket_composition_size | 1.107 | 0.356 | 0.350 | 0.006 |

All three are well calibrated (gap < 0.03), unlike the retrieval probe (gap 0.62). The ligand model's slope of 0.77 means it is mildly over-confident in its spread; the pocket models are very slightly under-confident.

**Note for later.** Two ways of averaging seeds give slightly different numbers: the training scripts report the *mean of per-seed r* (pocket composition 0.475), the analysis script reports *r of the mean prediction* (0.477). Both are defensible, they are not errors, and the report should state which it uses. I am treating the training-script CSVs as canonical for headline scores and the analysis tables as canonical for diagnostics.
