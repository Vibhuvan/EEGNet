"""
Data loading for the PhysioNet EEG Motor Movement/Imagery dataset (EEGBCI).

Task: left fist vs right fist motor imagery, runs [6, 10, 14] (per the
assignment spec). TRAIN_SUBJECTS = 1-8, TEST_SUBJECTS = 9-10.

Design decisions (documented, as required):
- Band-pass 8-30 Hz: covers mu (8-13Hz) and beta (13-30Hz) rhythms, which
  is where motor-imagery-related ERD/ERS lives. Filtering out DC drift and
  high-frequency muscle/line noise is standard practice for this dataset
  and is *not* a modeling choice we're evaluating — it's a fixed
  preprocessing step so that everything downstream isolates the
  spatial/temporal modeling question, not the filtering question.
- 2s windows starting at cue onset (T1/T2 events): matches the imagery
  cue duration in the PhysioNet protocol.
- We resample to 128 Hz to keep EEGNet's temporal kernel (window//2)
  reasoning tractable and to match the sampling rate EEGNet was
  originally tuned for.
"""
import os
import warnings
import numpy as np

FS = 128  # target sampling rate (Hz)
TMIN, TMAX = 0.0, 2.0  # seconds relative to event onset
LOW_FREQ, HIGH_FREQ = 8.0, 30.0  # mu+beta band
N_CHANNELS = 64  # standard EEGBCI montage
RUNS = [6, 10, 14]  # motor imagery: left vs right fist (per assignment)

# Cross-platform default data dir: <repo_root>/data (works on Windows/Mac/Linux)
DEFAULT_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

DATA_SOURCE = {"value": "unknown"}  # mutated to "real" or "synthetic"; read by callers to tag outputs
_REACHABLE_CACHE = {"checked": False, "value": False}  # cached so it's consistent across all load_subjects() calls in one run


def _physionet_reachable(timeout=6):
    """
    Fast pre-check so we don't burn minutes retrying downloads when the
    network is sandboxed. Cached at module level: this is checked ONCE per
    process and reused for every call, so train/test/reduced subjects can
    never end up on different data sources within the same run.
    """
    if _REACHABLE_CACHE["checked"]:
        return _REACHABLE_CACHE["value"]

    import urllib.request
    ok = False
    # one retry — a single transient timeout shouldn't doom the whole run to synthetic data
    for attempt in range(2):
        try:
            urllib.request.urlopen("https://physionet.org", timeout=timeout)
            ok = True
            break
        except Exception:
            continue
    _REACHABLE_CACHE["checked"] = True
    _REACHABLE_CACHE["value"] = ok
    return ok


def _try_load_real(subjects, runs, data_path):
    """Attempt to load real EEGBCI data via MNE. Returns list of (X, y) per subject, or None on failure."""
    if not _physionet_reachable():
        warnings.warn("physionet.org unreachable (network check failed) — "
                       "skipping download attempt and falling back to synthetic data.")
        return None

    import mne
    from mne.datasets import eegbci
    from mne.io import concatenate_raws, read_raw_edf

    mne.set_log_level("ERROR")
    os.makedirs(data_path, exist_ok=True)
    # pre-set the config so MNE never opens an interactive y/n prompt asking
    # to confirm the download path (that prompt blocks forever under a
    # non-interactive/piped stdin, and even interactively it's easy to miss)
    mne.set_config("MNE_DATASETS_EEGBCI_PATH", data_path, set_env=True)

    per_subject = []
    try:
        for subj in subjects:
            raw_fnames = eegbci.load_data(subj, runs, path=data_path, verbose=False)
            raws = [read_raw_edf(f, preload=True, verbose=False) for f in raw_fnames]
            raw = concatenate_raws(raws)
            eegbci.standardize(raw)
            raw.set_montage("standard_1005", on_missing="ignore")
            raw.filter(LOW_FREQ, HIGH_FREQ, fir_design="firwin", verbose=False)

            events, event_id = mne.events_from_annotations(raw, verbose=False)
            # T1 = left fist, T2 = right fist (rest / T0 excluded)
            wanted = {k: v for k, v in event_id.items() if k in ("T1", "T2")}
            if len(wanted) < 2:
                raise RuntimeError(f"Subject {subj}: expected T1/T2 events, got {event_id}")

            epochs = mne.Epochs(
                raw, events, event_id=wanted, tmin=TMIN, tmax=TMAX,
                baseline=None, preload=True, verbose=False,
            )
            epochs.resample(FS, verbose=False)
            X = epochs.get_data()  # (n_epochs, n_channels, n_times)
            y = np.array([0 if epochs.events[i, 2] == wanted["T1"] else 1
                           for i in range(len(epochs))])
            per_subject.append((X.astype(np.float32), y.astype(np.int64), subj))
        return per_subject
    except Exception as e:
        warnings.warn(f"Real PhysioNet download/load failed ({type(e).__name__}: {e}). "
                       f"Falling back to synthetic data — see src/data.py docstring.")
        return None


def _make_synthetic(subjects, runs, n_trials_per_class=30, seed=42):
    """
    Physiologically-motivated synthetic fallback (NOT real data).

    Generates band-limited (mu/beta) signal with a lateralized amplitude
    decrease (ERD) over one hemisphere depending on class, injects 1/f
    pink noise, and applies a *per-subject random spatial mixing matrix*
    to emulate inter-subject head-geometry variability — this is what
    lets the synthetic set still exhibit a cross-subject generalization
    gap, so Part 1's cross-subject experiment is meaningful as a pipeline
    smoke test (though the *magnitude* of any effect is not to be trusted).
    """
    rng = np.random.default_rng(seed)
    n_times = int((TMAX - TMIN) * FS)
    t = np.arange(n_times) / FS
    per_subject = []
    # crude 64-ch layout indices for "left" / "right" motor cortex ROI
    left_roi = rng.choice(N_CHANNELS, size=8, replace=False)
    right_roi = rng.choice([c for c in range(N_CHANNELS) if c not in left_roi], size=8, replace=False)

    for subj in subjects:
        subj_rng = np.random.default_rng(seed + subj * 97)
        # per-subject random invertible spatial mixing (head-geometry proxy)
        mix = subj_rng.normal(0, 1, size=(N_CHANNELS, N_CHANNELS))
        mix += np.eye(N_CHANNELS) * 2.0  # keep it well-conditioned
        n_total = n_trials_per_class * 2
        X = np.zeros((n_total, N_CHANNELS, n_times), dtype=np.float32)
        y = np.zeros(n_total, dtype=np.int64)
        for i in range(n_total):
            cls = i % 2
            y[i] = cls
            base = np.zeros((N_CHANNELS, n_times))
            for ch in range(N_CHANNELS):
                mu_amp = subj_rng.uniform(0.5, 1.5)
                beta_amp = subj_rng.uniform(0.2, 0.6)
                phase = subj_rng.uniform(0, 2 * np.pi)
                sig = (mu_amp * np.sin(2 * np.pi * 10 * t + phase)
                       + beta_amp * np.sin(2 * np.pi * 20 * t + phase * 0.5))
                base[ch] = sig
            # ERD: suppress amplitude in the contralateral ROI for the imagined hand
            erd_roi = right_roi if cls == 0 else left_roi  # left fist -> right hemisphere ERD
            base[erd_roi] *= 0.4
            # pink-ish noise
            noise = subj_rng.normal(0, 0.8, size=(N_CHANNELS, n_times))
            trial = base + noise
            trial = mix @ trial  # per-subject spatial mixing
            X[i] = trial.astype(np.float32)
        # z-score per channel
        X = (X - X.mean(axis=2, keepdims=True)) / (X.std(axis=2, keepdims=True) + 1e-6)
        per_subject.append((X, y, subj))
    return per_subject


def load_subjects(subjects, runs=RUNS, data_path=DEFAULT_DATA_PATH):
    """
    Returns list of (X, y, subject_id) tuples.
    X: (n_trials, n_channels, n_times) float32
    y: (n_trials,) int64, 0=left fist, 1=right fist
    """
    real = _try_load_real(subjects, runs, data_path)
    if real is not None:
        DATA_SOURCE["value"] = "real"
        return real
    DATA_SOURCE["value"] = "synthetic"
    print("=" * 70)
    print("WARNING: using SYNTHETIC fallback data (PhysioNet unreachable).")
    print("Results from this run are a pipeline smoke test ONLY.")
    print("Re-run on a machine with internet access to physionet.org")
    print("before reporting any numbers.")
    print("=" * 70)
    return _make_synthetic(subjects, runs)


def stack(per_subject):
    """Concatenate a list of (X, y, subj) into arrays, plus a subject-id array."""
    Xs, ys, subj_ids = [], [], []
    for X, y, subj in per_subject:
        Xs.append(X)
        ys.append(y)
        subj_ids.append(np.full(len(y), subj, dtype=np.int64))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(subj_ids)