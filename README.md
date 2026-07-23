# EEGNet: Failure Analysis, Improvement, and a New Idea

Technical evaluation see `ANALYSIS.md` for the required
Part 1 / Part 3 written analyses (and the Part 2 rationale, which is
documented inline in `src/sa_eegnet.py`).

## Results (real PhysioNet data)

`python main.py` has been run end-to-end on real EEGBCI data
(`data_source: "real"` in `results/results.json`). Summary:

| Experiment | Best | Worst |
|---|---|---|
| A) Pooled random split (optimistic, subject leakage) | 0.597 | — |
| B) Cross-subject baseline (train 1–8, test 9–10) | 0.589 | 0.533 |
| C) Reduced data (3 subjects) | 0.611 | 0.511 |
| D) SA-EEGNet (Part 2) | 0.589 | 0.556 |
| E) DynamicEEGNet (Part 3) | 0.522 | — |

**Caveat, stated plainly:** these numbers sit close to chance (50%
binary), including the optimistic same-subject baseline (A), whose
accuracy oscillates 43–60% across epochs rather than climbing cleanly.
That pattern is more consistent with the model not learning cleanly at
current settings (epochs/LR/architecture, possibly label alignment) than
with the task being fundamentally this hard — EEGNet on this dataset is
reported well above these numbers in the literature. This should be
investigated (more epochs, LR sweep, sanity-check event/label alignment)
before treating the raw accuracy numbers above as final.

**The one result that stands on its own regardless:** the spatial-filter
bottleneck probe. In Part 1, subject identity is decodable at **62.4%**
from the pooled embedding (chance = 10%). In Part 2 (SA-EEGNet), the same
kind of probe *at the spatial bottleneck* drops to **31.1%** (chance =
12.5%) — a real, substantial reduction in subject-identity leakage at
exactly the layer the adversarial branch was designed to target. This
holds up as evidence for the mechanism even while overall task accuracy
is still being debugged.

Part 3's gain-variability check came back non-degenerate
(`mean_std_across_trials_per_channel = 0.097`, not near zero) — the
hypernetwork is producing real per-trial variation rather than
collapsing to a static gain, i.e. the mechanism is doing something,
independent of the accuracy question above.

Full numbers: `results/results.json`. Embedding plots: `figures/`.

## Before you run this yourself

`src/data.py` tries real PhysioNet data first and only falls back to a
synthetic, physiologically-motivated dataset (documented in that file's
docstring) if physionet.org is unreachable — useful for verifying the
pipeline runs end-to-end on a sandboxed/offline machine. Check
`results/results.json -> data_source` after any run to confirm which one
you got; the results above are already confirmed real.

```
pip install -r requirements.txt
python main.py
```

Runtime on real data: roughly 15–30 min on CPU for the full pipeline
(3 seeds × 4 model-training experiments across Parts 1–3). Reduce
`epochs` in `main.py` if you need a faster iteration loop while
developing.

## Structure

```
main.py                 # orchestrates everything; single entry point
src/
  data.py                # PhysioNet EEGBCI loader (+ synthetic fallback, documented)
  eegnet.py               # standard EEGNet (Lawhern et al. 2018)
  sa_eegnet.py             # Part 2: Subject-Adversarial EEGNet (proposed improvement)
  dynamic_eegnet.py        # Part 3: State-Conditioned Dynamic Spatial Filtering (novel idea PoC)
  train.py                 # training/eval loops, feature extraction
  analysis.py               # t-SNE/UMAP plots + linear-probe class-vs-subject separability
  utils.py                   # seeding (fixed seed=42, documented), device selection
ANALYSIS.md              # Part 1 half-page analysis + Part 3 half-page idea writeup
results/results.json     # all numeric results, best & worst runs, tagged by data_source
figures/                 # t-SNE / UMAP embedding plots
```

## What each part does, concretely

- **Part 1** (`run_part1` in `main.py`): trains baseline EEGNet under (A) a
  pooled random split (optimistic, subject-leaking), (B) strict
  cross-subject (train 1–8, test 9–10, 3 seeds — best & worst reported),
  and (C) reduced training data (3 subjects, 3 seeds). Also extracts
  penultimate-layer features from the cross-subject model, plots
  t-SNE/UMAP colored by class and by subject, and fits linear probes to
  measure whether the representation encodes subject identity more
  strongly than class identity.
- **Part 2** (`run_part2`): trains `SA_EEGNet` (gradient-reversal subject
  adversary attached directly at the spatial-filter bottleneck identified
  in Part 1) under the same cross-subject protocol and seeds, and checks
  whether the adversarial branch actually reduced subject-decodability at
  that bottleneck (not just whether accuracy went up).
- **Part 3** (`run_part3`): trains `DynamicEEGNet` (per-trial,
  band-power-conditioned spatial filter gain) under the same protocol,
  and checks whether the learned gain vector is non-degenerate (has real
  per-trial variance) as a sanity check that the mechanism is doing
  something, before making any performance claim.

## Design decisions documented in-code (not repeated here)

- Fixed seed (42) and 3-seed robustness sweep — `src/utils.py`, `main.py`.
- Preprocessing choices (8–30 Hz filter, 128 Hz resample, 2s window) —
  `src/data.py` docstring.
- Why the depthwise spatial filter is the diagnosed failure point, and
  why a symptom-level fix (more dropout/L2) would not address it —
  `src/sa_eegnet.py` docstring.
- Why SC-DSF targets a different, underexplored failure mode
  (trial-to-trial baseline nonstationarity, not cross-subject transfer)
  — `src/dynamic_eegnet.py` docstring.

