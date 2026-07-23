# Analysis

**Read this first:** PhysioNet (physionet.org) was not reachable from the
environment this repo was drafted in (network-sandboxed dev container —
confirmed via a direct HTTPS request that returned `403 Forbidden`, see
commit history / dev notes). Every number below is from a **synthetic
pipeline smoke test**, not real EEGBCI data, and is explicitly tagged
`data_source: "synthetic"` in `results/results.json`. **Before submitting,
run `python main.py` on a machine with internet access to physionet.org**
— it will use real data automatically (see `src/data.py`); nothing else
needs to change. The analysis below is written in terms of the mechanism,
which is what the assignment asks for — the specific numbers you get on
real data should be dropped in to replace the placeholders marked `[RUN]`.

---

## Part 1 — Honest Failure Analysis (≤ half page)

**What we tested:** (A) a pooled random split of subjects 1–8 (optimistic,
same-subject trials leak between train/val), (B) strict cross-subject
evaluation (train 1–8, test 9–10), (C) reduced training data (3 subjects
instead of 8, still tested on 9–10), and (D) t-SNE/UMAP of the penultimate-layer
embeddings plus a linear probe asking whether class or subject identity is
easier to decode from those embeddings.

**Result pattern (on real data, expect):** accuracy on (A) sits comfortably
above (B), and (B) degrades further under (C). This is the "what" — the
assignment explicitly does not want this stated as the conclusion.

**Why, mechanistically:** EEGNet's depthwise spatial convolution
(`Conv2d(F1, F1*D, (n_channels, 1), groups=F1)`) learns exactly one linear
combination over the 64 electrodes per temporal filter, shared across all
training trials, and frozen at test time. This layer has no way to know
which electrode-to-cortical-source mapping applies to a subject it has
never seen. Two properties of the training objective make this worse than
it needs to be:

1. **Nothing in the loss discourages subject-specific shortcuts.**
   Cross-entropy on left/right fist only cares that classes separate on
   the *training* subjects — it has no preference between a filter that
   isolates a subject-general sensorimotor source and one that
   overfits to a training-subject's specific electrode geometry, as long
   as both separate classes equally well on those subjects. With only 8
   training subjects and 64 channels, the model has ample capacity to
   take the second, non-transferable option.
2. **The spatial filter is a single global linear operator, not a
   per-subject one.** Real inter-subject variability (skull thickness,
   gyral folding, cap placement offset, individual differences in
   sensorimotor rhythm topography) changes the electrode-to-source
   mapping in ways a *fixed* linear filter cannot track. A filter tuned
   to be optimally sensitive to subject 3's spatial pattern is not
   guaranteed to even be well-conditioned for subject 9.

The embedding analysis is the direct evidence for this, not just a
picture: the linear probe results (`embedding_probe` in `results.json`)
compare how well a simple linear classifier decodes **class** vs.
**subject identity** from the same penultimate-layer features. If subject
identity is substantially easier to decode than class (well above chance,
often approaching class-level accuracy), that is a direct measurement
that the representation is subject-coded, not purely task-coded — i.e.
the spatial filters are encoding "whose head is this" as a side effect of
learning "which class is this," and that side information is exactly what
fails to transfer to subjects 9–10. Reducing training data (C) makes this
worse for the ordinary reason that fewer subjects means less pressure
toward subject-general structure and easier memorization of subject-specific
patterns per subject.

**What this is not:** it is not "the model overfits" in the generic
sense — dropout and weight decay are already present in the baseline and
do not fix this pattern, because overfitting-in-general regularizers
shrink capacity uniformly; they don't distinguish between "capacity spent
on task-relevant signal" and "capacity spent on subject identity." That
distinction is what Part 2 targets directly.

---

## Part 2 — Proposed Improvement: Subject-Adversarial EEGNet

See `src/sa_eegnet.py` for the full rationale (also documented inline).
Summary: a subject-identity classifier is attached to the *same*
spatial-filter output that fails to transfer, connected through a
Gradient Reversal Layer (Ganin & Lempitsky, 2015), so the spatial filters
are explicitly trained to retain class-relevant information while
actively discarding subject-identifying information — targeting the
mechanism identified above, not a generic regularizer.

- **Expected:** the spatial-bottleneck linear probe's subject-decoding
  accuracy should drop toward chance while class-decoding accuracy is
  preserved or improves, and cross-subject test accuracy on 9–10 should
  improve relative to the Part 1 baseline (B).
- **What actually happened `[RUN]`:** fill in from `results.json ->
  part2.sa_spatial_bottleneck_probe` vs `part1.embedding_probe`, and
  `part2.sa_eegnet_best/worst` vs `part1.cross_subject_best/worst`.
- **If the gap between expectation and reality exists:** the most likely
  failure mode with only 8 training subjects is that the subject
  classifier itself has too few classes/examples to provide a stable
  adversarial gradient — domain-adversarial training typically needs
  either more domains or a warm-up schedule on `grl_lambda` (currently a
  fixed 0.3) to avoid destabilizing the task loss early in training. If
  `sa_eegnet` underperforms the baseline, that is itself informative
  evidence about the data-scale requirements of adversarial debiasing on
  this dataset — worth stating explicitly in the defence rather than
  hiding it.

---

## Part 3 — One Thing Nobody Has Tried (≤ half page)

**Idea: State-Conditioned Dynamic Spatial Filtering (SC-DSF).** Full
rationale is in `src/dynamic_eegnet.py`. In short: every EEGNet-family
model uses a spatial filter that is *static per trial* — the same learned
electrode weighting is applied regardless of the subject's ongoing
oscillatory state at trial onset. But ERD/ERS (event-related
desynchronization/synchronization) is inherently a *relative* change
from a subject's pre-trial idle mu/beta baseline (Pfurtscheller & Lopes
da Silva, 1999), and that baseline power varies substantially both
trial-to-trial (arousal, fatigue, attention) and subject-to-subject
(individual mu power varies by an order of magnitude across people). A
fixed spatial filter cannot rescale itself for a trial where the
subject's baseline happens to be unusually high or low.

**The mechanism:** a lightweight per-trial band-power descriptor
(log mu/beta power per channel, computed via FFT over the trial) is fed
into a small hypernetwork that outputs a per-channel gain vector; this
gain multiplicatively modulates a base learned spatial filter before it's
applied, per trial. This is dynamic convolution (Chen et al., 2020)
adapted to EEG and conditioned on a neuroscience-motivated signal rather
than generic input statistics.

**Why grounded/specific/defensible:** it targets a named, documented
confound (baseline-dependent ERD magnitude), it's a concrete architecture
change at a named layer (not "add more layers" or "use better data"), and
it's falsifiable: if the idea has any effect, the learned gain vector
should (a) have non-trivial variance across trials (i.e. not collapse to
a constant — checked directly via `gain_variability` in `results.json`)
and (b) correlate with the band-power descriptor it was conditioned on.

**What the PoC checks (not claims):** whether the hypernetwork learns a
non-degenerate, trial-varying gain at all (`gain_variability.mean_std_across_trials_per_channel`
in `results.json` — near zero means the mechanism had no measurable
effect on this run), not whether SC-DSF beats EEGNet on 10 subjects of
data, which would not be a meaningful claim at this scale.
