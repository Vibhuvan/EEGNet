"""
Utilities: reproducibility, device selection, and a synthetic-data fallback.

Why the synthetic fallback exists (read this):
PhysioNet (physionet.org) is not reachable from every network environment
(institutional firewalls, sandboxed CI runners, etc.). We do not want
`python main.py` to hard-crash in that situation with no way to verify the
rest of the pipeline is correct. So: if the real EEGBCI download fails,
`data.py` falls back to a *physiologically-motivated synthetic* generator
(band-limited oscillations with a mu/beta ERD-like event over C3/C4, plus
per-subject spatial mixing to mimic head-geometry variability) purely so
every downstream stage (training, cross-subject eval, t-SNE/UMAP, Part 2,
Part 3) can be exercised and unit-checked.

IMPORTANT: synthetic numbers are NOT scientific results and must never be
reported as such. Every place that touches synthetic data prints a loud
warning and results are tagged with `data_source: "synthetic"` in the
saved JSON so it's unambiguous downstream. Run this on a machine with
PhysioNet access to get the real numbers this assignment asks for.
"""
import os
import random
import numpy as np
import torch

SEED = 42  # fixed & documented, per assignment requirement


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic cuDNN (slower, but reproducible — we care more about
    # reproducibility than speed on a dataset this small)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)
