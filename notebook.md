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

---

## 2026-10-12 — Graph Representation and Equivariant model implementation

**Built:**

- `src/lba/data/graph.py` — complex -> PyG graph. Hydrogens dropped (consistent with both baselines: they are ~52% of atoms and computationally added). 8 $\AA$ pocket trim around the ligand, 5 $\AA$ radius edges. Node scalars are element one-hot + is-ligand flag. No node vectors, since a lone atom has no intrinsic orientation. Edge scalars are 16 Gaussian RBFs over distance + 3 edge-type flags (pocket-pocket / ligand-ligand / pocket-ligand). Edge vectors are unit directions. Coordinates never enter as features. Whole splits cache to `.pt` keyed on a hash of the data root and both cutoffs.

- `src/lba/models/gvp.py` — GVP-GNN, 571,271 parameters. Written from the papers
  rather than ported from `drorlab/gvp-pytorch`, which targets torch 1.x. Three
  message-passing rounds, hidden dims (100 scalar, 16 vector), vector gating,
  equivariant LayerNorm (divide by RMS vector norm) and whole-channel vector
  dropout. Mean aggregation, not sum, deliberately: node degree tracks pocket
  size, which is the documented confound here, and sum would hand the model that
  confound as a magnitude proportional to degree.

Graph sizes: train 3507 (median 204 nodes / 3694 edges, max 492 nodes), val 466
(187 / 3309), test 490 (196 / 3433).

---

## 2026-10-14 — Equivariance testing and fixing the graph-construction bug

**Built:**
- `scripts/test_equivariance.py` — the gate. Rotation, translation and permutation arms, in float32 and float64.

- `scripts/train_gvp.py` — training, with per-epoch resumable checkpointing and an `--overfit N` sanity mode.

- `scripts/analyze_gvp.py` — paired comparison against every baseline.


**Equivariance passes exactly.** End-to-end float64, maximum absolute change in
predicted pKd over 12 complexes:

| Check | float32 | float64 |
|---|---:|---:|
| rotation | 2.98e-08 | 5.55e-17 |
| translation | 2.98e-08 | 6.94e-17 |
| permutation | 4.47e-08 | 8.33e-17 |

Machine epsilon. Tested with untrained weights on purpose, invariance is a
property of the architecture, not of the fit, and testing a checkpoint would
conflate the two.

**Bug** First run: float32 passed, float64 failed with rotation 7.9e-6 and permutation 2.5e-5. Permutation failing while rotation was comparatively clean pointed at indexing, not geometry.

Cause: `torch_cluster.radius_graph` truncates to `max_num_neighbors` **by atom index order, not by distance**.
So a cap that actually engages discards arbitrary close contacts, and *which* contacts it discards depends on the order atoms happen to appear in the deposited frame. I had set 32, a conventional default.
Measured over 300 training complexes:

| | |
|---|---:|
| uncapped max degree at 5 $\AA$ | 42 |
| complexes with a node above a 32-cap | **77%** |
| complexes with a node above a 48-cap | 0% |

Set the cap to 48. Median edges rose 3412 -> 3694, about +9%. Nothing is
truncated now, so the graph is a true radius graph and does not depend on atom
ordering.

This is the kind of bug that produces a plausible number from wrong data. No
loss curve would have shown it. Note for the report: the roadmap only called for
a rotation test; the permutation arm is what caught this. **If `edge_cutoff`
ever goes above 5 $\AA$, the degree ceiling has to be re-measured.**

---

## 2026-10-15 - Debugging and validating training pipeline

**Bug** The overfit sanity check plateaued at training loss 6.0 on 16 complexes and val r ~0.76. Predicting the mean of that subset gives MSE 6.09 — so it had learned essentially nothing, on data it was looking at directly. A 571k-parameter model must be able to memorise 16 examples.

Diagnosed properly instead of guessing, with a throwaway script isolating
optimisation from architecture:

| Configuration | lr | final loss | r |
|---|---:|---:|---:|
| as-built, no scheduler, 600 steps | 1e-3 | 0.0000 | +1.000 |
| as-built, no scheduler, 600 steps | 3e-3 | 0.0000 | +1.000 |
| as-built, no scheduler, 600 steps | 1e-2 | 6.0949 | −0.082 |

So the architecture was fine and the training loop was not. Two errors, both
mine: `ReduceLROnPlateau(patience=5)` was firing on validation r, which in
overfit mode is the training metric and is noisy and non-monotonic early, it
had cut the LR to 6.25e-5 by epoch 32, before convergence was possible. And in
overfit mode one epoch is one optimiser step, so `--epochs 120` gave 120 steps
against the ~600 actually needed.

Fixed: `--lr-patience 0` now disables the scheduler and the overfit test uses
it; overfit budget raised to 800 steps; default patience raised 5 → 10 for real
runs. That last change matters most — in a real run the same premature decay
would have looked like a converged model rather than a strangled one, with no
known-correct answer to compare against. After the fix: loss 52.4 → 0.0000,
r = +1.000. **Lesson: a sanity check needs a stated pass threshold *and* a
stated reference value, or a plateau at predict-the-mean looks like a loss.**

Also learned: lr 1e-2 diverges outright. Usable band is 1e-3 to 3e-3.

**One architectural measurement worth keeping.** At initialisation, pairwise
cosine similarity between the pooled graph representations is **0.9943** (min
0.9759) — per-node LayerNorm followed by mean-pooling really does squash the
graphs together. Training overcomes it on 16 examples, so it is not fatal, but
if the full runs train sluggishly this is the first suspect, and pooling before
the final norm is the obvious thing to try.

---

## 2026-10-16 - First experiment and preliminary results

**First real result - seed 0.** GTX 1650 Ti, 4.3 GB, batch 16, 45 s/epoch, no
OOM. Early-stopped at epoch 33, best epoch 18 (val r +0.559). Test r = **+0.520**,
RMSE 1.460. Local training is viable, Colab is insurance, not a blocker.

**Paired bootstrap, seed 0 only — provisional.** 3000 resamples, 490 test
complexes, joined on `pdb_id` with a `y_true` agreement check:

| GVP vs | Δr (all, n=490) | 95% CI | Δr (novel, n=278) | 95% CI |
|---|---:|---|---:|---|
| ligand-only | +0.004 | [−0.047, +0.055] | +0.007 | [−0.062, +0.074] |
| pocket, size-free | +0.043 | [−0.031, +0.114] | +0.004 | [−0.095, +0.106] |
| pocket + size | −0.077 | [−0.150, −0.007] | −0.134 | [−0.237, −0.031] |
| n_residues (one integer) | −0.056 | [−0.123, +0.008] | −0.112 | [−0.211, −0.011] |

**Concluded (provisionally).**

- Volkov replicated a third time, now with my own interaction-aware model: GVP
  and ligand-only differ by +0.004 and cannot be separated. A model given the
  full complex geometry does not beat one that never sees the protein.
- GVP loses significantly to pocket+size, and on novel ligands it loses
  significantly to `n_residues` — a linear fit on one integer. The novel stratum
  is where interaction modelling should matter most, because it is where ligand
  identity stops being available as a shortcut. That is the strongest version of
  the benchmark critique I have.
- "Not resolvable" is the finding, not a null to explain away. It is evidence
  the benchmark does not distinguish these models.

**Caveats, to deal with before any of this goes in the report.**

1. **Seed 0 only.** Seeds 1 and 2 are still running. The seed-mean prediction is a
   3-model ensemble and will probably score a little higher, which could move
   the novel-stratum interval against `n_residues` (currently clearing zero by
   only 0.011) to straddle zero. 
2. **0.520 looks low.** Untuned, single configuration, 60-epoch budget, 8 $\AA$ trim, and the selected epoch was a noisy val peak (val r bounced 0.499–0.559, test came in below val). A reviewer will say the interaction model is undertrained. Jing et al. report RMSE only for LBA (1.594), which is beaten by a mode lthat never sees the protein, and by a linear fir on residue count.
3. State the training budget plainly in Methodology.

**TODO.**
- Re-run `scripts/analyze_gvp.py` once seeds 1–2 land, and update the table above.
- Start with the generation of negatives.

---

## 2026-10-20 - Three-seed evaluation and a correction of the conclusions

**Supersedes the provisional paired table in the entry above.** That table was
seed 0 only, and seed 0 turned out to be the worst of the three. Leaving it in
place rather than editing it.

**Three seeds.** 60-epoch budget, early-stop patience 15, same protocol for all
three.

| Seed | Best epoch | Val r | Test r |
|---|---:|---:|---:|
| 0 | 18 | +0.559 | +0.520 |
| 1 | 44 | +0.660 | +0.614 |
| 2 | 47 | +0.632 | +0.527 |

Mean of per-seed r = **0.554** (sd 0.043). r of the seed-mean prediction =
**0.596**. The gap between those two is 0.042 — far wider than the 0.475/0.477
discrepancy noted for the pocket baselines on 2026-09-29, because averaging
three neural nets is a real ensemble while averaging three 500-tree forests is
nearly a no-op.

**Seed 0 stopped prematurely.** It peaked at epoch 18 and stopped at 33; seeds 1
and 2 peaked at 44 and 47. Val r moves +-0.03 between adjacent epochs, so
patience 15 is small relative to the noise. Two of three seeds were still
finding their best near the 60-epoch ceiling, so the budget is truncating them,
not just seed 0.

**Paired bootstrap on the seed-mean prediction** (3000 resamples, 490 complexes):

| GVP vs | delta r (all) | 95% CI | delta r (novel, n=278) | 95% CI |
|---|---:|---|---:|---|
| ligand-only | +0.080 | [+0.033, +0.129] | +0.078 | [+0.013, +0.146] |
| pocket, size-free | +0.119 | [+0.056, +0.179] | +0.075 | [−0.007, +0.159] |
| pocket + size | −0.001 | [−0.064, +0.061] | −0.063 | [−0.151, +0.028] |
| n_residues | +0.020 | [−0.043, +0.080] | −0.040 | [−0.127, +0.053] |

Read alone, this says the interaction model significantly beats both blinded
baselines — which would partly refute the premise of the project. It does not
survive scrutiny.

**The ensemble comparison is asymmetric, and that was the trap.** The GVP column
is the mean of three networks; the baselines are random forests, already
ensembles of 500 trees, so seed-averaging them changes almost nothing. Added
per-seed paired bootstraps to `scripts/analyze_gvp.py` — one network against one
forest, which is the like-for-like test:

| GVP vs | seed 0 | seed 1 | seed 2 | seeds winning |
|---|---:|---:|---:|---|
| ligand-only | +0.004 tie | +0.098 win | +0.011 tie | 1/3 |
| pocket, size-free | +0.043 tie | +0.138 win | +0.050 tie | 1/3 |
| pocket + size | −0.077 LOSS | +0.018 tie | −0.070 LOSS | 0/3 |
| n_residues | −0.056 tie | +0.038 tie | −0.049 tie | 0/3 |

The ensemble win is ensembling plus one outlier seed. Seeds 0 and 2 differ from
ligand-only by +0.004 and +0.011, which is nothing.

**Concluded.**

- **Volkov replicated, with my own interaction-aware model.** A single
  E(3)-equivariant network over the full complex graph is statistically
  indistinguishable from a 217-descriptor model that never sees the protein
  (2 of 3 seeds tie; deltas +0.004 and +0.011).
- **GVP never beats `n_residues`.** 0 of 3 seeds win, all three ties. An
  equivariant interaction model, verified rotation-invariant to machine epsilon,
  cannot be distinguished from a linear fit on the number of residues in the
  pocket — a feature that exists only because ATOM3D cut each pocket around a
  known ligand pose, and which could not be computed in the real prospective
  task. This is the central result of the report.
- **GVP loses to pocket+size in 2 of 3 seeds.** The confound beats the deep
  model outright.
- The ensemble result (+0.080 over ligand-only, significant) goes in the report
  as a **sensitivity analysis, not a headline**, with the asymmetry stated: it
  reflects ensembling against an already-ensembled baseline, and individual
  networks do not reproduce it.

**Methodological note for the report.** The headline direction depends on which
seed-averaging convention is used: mean of per-seed r gives 0.554 (below
`n_residues` at 0.576), r of the seed-mean gives 0.596 (above it). Per the
2026-09-29 convention, the training-script CSVs are canonical, so 0.554 is the
reported number. The report must state this explicitly rather than pick the
favourable one; a reader who recomputes it the other way and finds the sign
flipped will not be charitable. Both conventions agree qualitatively: a
statistical tie with counting residues.

**TODO.**
- Re-run all three seeds at 120 epochs, patience 30. Two of three were still
  improving at the ceiling and sd 0.043 is larger than several deltas I want to
  report. 
- Start with the generation of negatives.

---

## 2026-10-21 — 120 epochs: the budget was not the problem

**What I ran.** All three seeds at `--epochs 120 --early-stop 30`. Everything
else identical to the previous entry: batch 16, lr 1e-3, lr-patience 10, 8 $\AA$
pocket trim, 5 $\AA$ edge cutoff, same three seeds, same local GTX 1650 Ti. Only the
budget and the early-stopping patience changed, so any difference is
attributable to those two numbers (or to noise, which turns out to matter more
than I expected).

**The budget is no longer binding.** This was the whole point of the re-run.

| Seed | Epochs run | Best epoch | Stopped because |
|---|---:|---:|---|
| 0 | 86 | 55 | patience 30 |
| 1 | 75 | 44 | patience 30 |
| 2 | 68 | 37 | patience 30 |

All three early-stopped well short of the 120 ceiling, and no best epoch is anywhere near it. The truncation I diagnosed last time is gone. The models are not undertrained any more.

**Results, against the 60-epoch run.**

| | 60 epochs | 120 epochs |
|---|---:|---:|
| seed 0 | best 18 · val +0.559 · test **+0.520** · RMSE 1.460 | best 55 · val +0.604 · test **+0.532** · RMSE 1.501 |
| seed 1 | best 44 · val +0.660 · test **+0.614** · RMSE 1.505 | best 44 · val +0.610 · test **+0.518** · RMSE 1.682 |
| seed 2 | best 47 · val +0.632 · test **+0.527** · RMSE 1.567 | best 37 · val +0.598 · test **+0.494** · RMSE 1.594 |
| mean of per-seed test r | **0.554** | **0.515** |
| r of the seed-mean prediction | 0.596 | 0.550 |
| mean test RMSE | 1.511 | 1.593 |

**Doubling the budget made test performance slightly worse, not better.** Mean
per-seed r fell from 0.554 to 0.515 while validation r rose for seed 0 and
tightened across seeds (0.598-0.610, spread 0.012 against 0.101 before). More
training bought a better-looking validation peak and a slightly worse test
score. That is epoch selection chasing validation noise, not a model that needed
more time.

**What this settles.** Caveat 2 from the 2026-10-16 entry anticipated the
obvious reviewer objection: 0.520 is at the low end of the published range, the
model is undertrained, run it longer. After running it twice as long with a patience
twice as forgiving, every seed converged and stopped on its own, and the number
went *down*. That objection is now closed, and the central claim survives it:

> An E(3)-equivariant model over the full complex graph, verified
> rotation-invariant to machine epsilon and trained to convergence, scores
> r = 0.515 on the ID30 test split: statistically indistinguishable from a
> 217-descriptor model that never sees the protein (0.516) and from a linear
> fit on pocket residue count (0.576), and significantly *worse* than both
> the residue count and pocket+size on the novel-ligand stratum.

**Paired bootstrap, 3000 resamples, 490 complexes, joined on `pdb_id`.** On the
seed-mean prediction - the per-seed table below is the like-for-like one.

| GVP vs | delta r (all, n=490) | 95% CI | delta r (novel, n=278) | 95% CI |
|---|---:|---|---:|---|
| ligand-only | +0.034 | [-0.018, +0.086] | -0.008 | [-0.083, +0.066] |
| pocket, size-free | **+0.073** | [+0.007, +0.135] | -0.011 | [-0.101, +0.081] |
| pocket + size | -0.047 | [-0.114, +0.020] | **-0.149** | [-0.243, -0.051] |
| n_residues | -0.026 | [-0.094, +0.039] | **-0.126** | [-0.222, -0.024] |

Per-seed, one network against one forest, no ensembling:

| GVP vs | seed 0 | seed 1 | seed 2 | seeds winning |
|---|---:|---:|---:|---|
| ligand-only | +0.016 tie | +0.002 tie | -0.023 tie | 0/3 |
| pocket, size-free | +0.056 tie | +0.042 tie | +0.017 tie | 0/3 |
| pocket + size | -0.064 tie | -0.078 LOSS | -0.103 LOSS | 0/3 |
| n_residues | -0.044 tie | -0.058 tie | -0.082 LOSS | 0/3 |

**The 60-epoch ensemble win is gone.** At 60 epochs the seed-mean beat
ligand-only by +0.080 [+0.033, +0.129], and I flagged it as a sensitivity
analysis rather than a headline on the grounds that it was three networks
averaged against an already-ensembled forest. At 120 epochs the same comparison
is +0.034 [-0.018, +0.086] - not resolvable. The caveat was right and the effect
did not reproduce. Nothing in the report rests on it.

**The novel-ligand stratum is now a significant loss, and this is the result of
the whole project so far.** At 60 epochs, seed-mean against `n_residues` on
novel ligands was -0.040 [-0.127, +0.053], a tie. At 120 epochs it is **-0.126
[-0.222, -0.024]** and the interval clears zero. Against pocket+size it is
**-0.149 [-0.243, -0.051]**. On the stratum where ligand identity stops being
available as a shortcut, and where interaction modelling ought therefore to
matter most, the converged equivariant model is *significantly worse* than a
linear fit on the number of residues in the pocket - a feature that exists only
because ATOM3D cut each pocket around a known ligand pose, and which could not
be computed in the real prospective task.

**The one win, and why it rescues nothing.** GVP does beat the size-free pocket
model overall, +0.073 [+0.007, +0.135], the only resolvable win anywhere in the
table. But the same comparison on novel ligands is -0.011, a tie. The advantage
exists only where ligand identity is available and evaporates where it is not.
That is the shortcut thesis restated, not a counterexample to it.

**My two scripts disagree on sd.** `analyze_gvp.py` prints 0.016 for the three
seeds, `train_gvp.py` prints 0.020 for the same three numbers - ddof=0 against
ddof=1, now inside my own pipeline rather than between my notes and my code.
Fix one of them before the experiment matrix multiplies the problem by six
configurations.

*Resolved 2026-11-22:* `analyze_gvp.py` now uses `np.std(..., ddof=1)`, so every
script in the pipeline reports the sample sd. Checked on the 120-epoch table:
0.020 from both.

**Retracting the Jing RMSE comparison.** I noted last week that Jing et al.
report RMSE only (1.594) and that my model beat it. At 120 epochs my mean test
RMSE is **1.593**. That is a dead heat, and with a spread of 0.090 across seeds
the comparison was never stable enough to call either way. The honest wording
for the report is "comparable to the published GVP-GNN", and the sentence that
actually carries weight is the one that was always stronger: *every blinded
baseline I have beats both of us on RMSE* - pocket+size 1.358, `n_residues`
1.387, ligand-only 1.465.

**Seed 1 has an R2 of 0.003.** Test RMSE 1.682 against 1.684 for predicting the
test mean, while correlating at r = +0.518. It ranks complexes reasonably and is
almost worthless in absolute terms - a pure calibration failure. Worth knowing
because it means r and RMSE are not measuring the same thing for this model, and
RMSE comparisons between runs are dominated by calibration drift rather than by
anything about interactions. Lead with r in the report; give RMSE alongside and
say why.

**Training is not reproducible, and I can now prove it.** With a fixed seed and a
strictly longer budget, the first 60 epochs should be identical and the best
validation r can only go up. For seeds 1 and 2 it went **down** - 0.660 to 0.610
and 0.632 to 0.598. The trajectories diverged. Two causes, and they separate
cleanly:

1. **A real bug in my resume path.** The checkpoint saves model, optimizer,
   scheduler, epoch and history, but *not* the RNG state, so `torch.manual_seed`
   restarts the shuffling and dropout sequence from the beginning on every
   resume. Seed 1 was resumed twice (epochs 13 and 52) during the GPU trouble
   below, so its trajectory is genuinely a different run after epoch 13.
2. **Nondeterminism that is not a bug.** Seed 2 was never resumed - no
   `resumed seed 2` line in the log - and diverged anyway. `torch_scatter` sums
   messages with `atomicAdd`; float addition is not associative and atomics
   complete in arbitrary order, so identical seeds drift apart over tens of
   epochs. This is expected behaviour for GNN training on GPU and I had not
   accounted for it.

**Inference, by contrast, is exactly reproducible.** The 60-epoch CSVs were
overwritten by this run before I had committed them. I rebuilt them from
`results/checkpoints/gvp/seed*/best.pt` with the new
`scripts/score_checkpoints.py`, and every number reproduced to three decimals
seven days later - val, test, r, rho, RMSE, MAE, all three seeds, with each
`best.pt` epoch matching the argmax of its `history.json`. So the graph cache,
the element vocabulary and the forward pass are stable, and the nondeterminism
lives in the backward pass. That is a precise enough statement to defend in the
report.

**Consequence for how I report this.** The 60-epoch and 120-epoch runs are *not*
clean replicates of each other, so the 0.039 drop in mean r cannot be attributed
to the budget alone - part of it is run-to-run noise of unmeasured size. The
defensible claim is the one that holds under both budgets: **GVP sits at or
below `n_residues` and level with ligand-only.**

**An sd convention I had inconsistent.** I recorded sd 0.043 for the 60-epoch
run, which is the population sd (numpy default, ddof=0). `train_gvp.py` prints
the sample sd (pandas default, ddof=1), which for the same three numbers is
0.053. Like for like: 60-epoch 0.053 / 0.043, 120-epoch 0.020 / 0.016. The
tightening is real either way, but I need one convention stated once, the same
way I pinned the seed-averaging convention on 2026-09-29. **Going with ddof=1**,
because that is what the canonical training-script CSVs print.

**Hardware episode, recorded because it will recur.** The run took 19 h instead
of 3. The GPU was pinned at a 10 W enforced power limit against its 35 W
default, P8, memory clock 405 MHz against 5001 MHz - a 12x drop that showed up
as a 12x slowdown, 530 s/epoch against the 45 s/epoch I measured on 2026-10-16
on the same card. Not thermal: 58 C with a 97 C slowdown threshold. `nvidia-smi`
reported 99% utilisation throughout, which is a liveness signal and not a
performance one. A reboot restored full clocks, then the cap returned after
about 50 minutes, and a second reboot plus disabling the AC idle timeouts
(`powercfg /change monitor-timeout-ac 0`, `standby-timeout-ac 0`) held for the
rest of the run. **If a run is inexplicably slow, check `clocks.mem` and
`nvidia-smi -q -d PERFORMANCE` before anything else.** Timings from this laptop
are not trustworthy and should not appear in the report as benchmarks.

**Concluded.**

- **The interaction model is converged and still does not beat the blinded
  baselines.** r = 0.515 (sd 0.020, ddof=1). 0 of 3 seeds win a single paired
  comparison against any of the four references. The undertraining defence is
  spent.
- **Volkov replicated a fourth time, now at convergence.** GVP against
  ligand-only is +0.034 [-0.018, +0.086] overall and -0.008 [-0.083, +0.066] on
  novel ligands. A model given the full complex geometry cannot be separated
  from one that never sees the protein.
- **GVP is significantly worse than `n_residues` and than pocket+size on the
  novel-ligand stratum** (-0.126 and -0.149, both intervals clearing zero).
  This is the headline of the report: the stratum where interaction modelling
  should matter most is the stratum where the interaction model loses outright
  to counting residues.
- **The conclusion is robust to the training budget**, which is a stronger claim
  than either run makes alone, and the one to put in the report.
- **The 60-epoch ensemble win did not reproduce.** My own caveat was correct;
  the effect was ensembling, not modelling.
- **The Jing RMSE comparison is a tie**, not a win. Retracted.
- **My three-seed spread conflates seed variance with run-to-run
  nondeterminism.** Until that is measured we need to be careful with the precision.

**TODO.**

- Save and restore RNG state (`torch.get_rng_state`, `np.random.get_state`) in
  the checkpoint.
- **Measure the nondeterminism**: run seed 0 twice, uninterrupted, same
  everything, and compare test r. ~1.5 h, and it tells me how many decimal
  places I am allowed to report.
- Then negatives. The model work is done, the budget question is closed and the
  benchmark critique now has a significant result behind it; everything after
  this is Phase 3.


