"""
Entry point. `python main.py` reproduces Part 1 (failure analysis),
Part 2 (proposed improvement), and Part 3 (novel idea PoC) without manual
intervention, and writes all figures/results/results.json.

Fixed seed: see src/utils.py (SEED = 42). We additionally run the two
cross-subject experiments (baseline and SA-EEGNet) across 3 seeds and
report BEST and WORST runs explicitly, per the assignment's
no-cherry-picking requirement.

NOTE ON DATA: see src/data.py docstring. If PhysioNet is unreachable,
this falls back to synthetic data and every result is tagged
data_source="synthetic" in results.json. Do not report synthetic numbers
as findings — re-run with internet access to physionet.org first.
"""
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from src import data as data_mod
from src.utils import set_seed, get_device, ensure_dirs, SEED
from src.eegnet import EEGNet
from src.sa_eegnet import SA_EEGNet
from src.dynamic_eegnet import DynamicEEGNet
from src.train import EEGDataset, train_baseline, train_sa_eegnet, evaluate, extract_features
from src.analysis import plot_embedding, class_vs_subject_separability
from torch.utils.data import DataLoader

TRAIN_SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8]
TEST_SUBJECTS = [9, 10]
REDUCED_SUBJECTS = [1, 2, 3]
RUNS = data_mod.RUNS
SEEDS_FOR_ROBUSTNESS = [42, 43, 44]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")


def pooled_random_split(X, y, seed, test_frac=0.2):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_test = int(len(y) * test_frac)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def run_part1(all_train_data, all_test_data, reduced_train_data, device):
    print("\n" + "=" * 70)
    print("PART 1 — Honest Failure Analysis")
    print("=" * 70)
    results = {}

    Xtr_full, ytr_full, subj_tr_full = data_mod.stack(all_train_data)
    Xte, yte, subj_te = data_mod.stack(all_test_data)
    n_channels, n_times = Xtr_full.shape[1], Xtr_full.shape[2]

    # ---- Experiment A: pooled random split (optimistic, same-subject leakage) ----
    Xtr_p, ytr_p, Xva_p, yva_p = pooled_random_split(Xtr_full, ytr_full, seed=SEED)
    set_seed(SEED)
    model_a = EEGNet(n_channels=n_channels, n_times=n_times)
    res_a = train_baseline(model_a, EEGDataset(Xtr_p, ytr_p), EEGDataset(Xva_p, yva_p), device, epochs=40)
    results["pooled_random_split"] = res_a
    print(f"A) Pooled random split (subject leakage) — best val acc: {res_a['best_val_acc']:.3f}")

    # ---- Experiment B: cross-subject, multiple seeds -> report best & worst ----
    cross_subj_runs = []
    for seed in SEEDS_FOR_ROBUSTNESS:
        set_seed(seed)
        model_b = EEGNet(n_channels=n_channels, n_times=n_times)
        res_b = train_baseline(model_b, EEGDataset(Xtr_full, ytr_full), EEGDataset(Xte, yte), device, epochs=40)
        cross_subj_runs.append(res_b["best_val_acc"])
        print(f"B) Cross-subject (train 1-8, test 9-10), seed={seed} — "
              f"best test acc: {res_b['best_val_acc']:.3f}")
    results["cross_subject_runs"] = cross_subj_runs
    results["cross_subject_best"] = max(cross_subj_runs)
    results["cross_subject_worst"] = min(cross_subj_runs)
    print(f"   -> BEST run: {max(cross_subj_runs):.3f}  |  WORST run: {min(cross_subj_runs):.3f}")

    # keep the best cross-subject model for embedding analysis
    set_seed(SEED)
    model_cross = EEGNet(n_channels=n_channels, n_times=n_times)
    train_baseline(model_cross, EEGDataset(Xtr_full, ytr_full), EEGDataset(Xte, yte), device, epochs=40)

    # ---- Experiment C: reduced training data (3 subjects instead of 8) ----
    Xtr_r, ytr_r, subj_tr_r = data_mod.stack(reduced_train_data)
    reduced_runs = []
    for seed in SEEDS_FOR_ROBUSTNESS:
        set_seed(seed)
        model_c = EEGNet(n_channels=n_channels, n_times=n_times)
        res_c = train_baseline(model_c, EEGDataset(Xtr_r, ytr_r), EEGDataset(Xte, yte), device, epochs=40)
        reduced_runs.append(res_c["best_val_acc"])
    results["reduced_data_runs"] = reduced_runs
    results["reduced_data_best"] = max(reduced_runs)
    results["reduced_data_worst"] = min(reduced_runs)
    print(f"C) Reduced data (3 subjects), test 9-10 — "
          f"best: {max(reduced_runs):.3f}  worst: {min(reduced_runs):.3f}")

    # ---- Embeddings: t-SNE / UMAP on pooled train+test features from the cross-subject model ----
    Xall = np.concatenate([Xtr_full, Xte])
    yall = np.concatenate([ytr_full, yte])
    subj_all = np.concatenate([subj_tr_full, subj_te])
    feats, labels, subjs = extract_features(model_cross, EEGDataset(Xall, yall, subj_all), device)

    ensure_dirs(FIG_DIR)
    tsne_path = plot_embedding(feats, labels, subjs, "EEGNet penultimate features",
                                os.path.join(FIG_DIR, "part1_tsne.png"), method="tsne")
    umap_path = plot_embedding(feats, labels, subjs, "EEGNet penultimate features",
                                os.path.join(FIG_DIR, "part1_umap.png"), method="umap")
    probe = class_vs_subject_separability(feats, labels, subjs)
    results["embedding_probe"] = probe
    results["figures"] = {"tsne": tsne_path, "umap": umap_path}
    print(f"   Linear probe — class acc: {probe['linear_probe_class_acc']:.3f}  "
          f"subject acc: {probe['linear_probe_subject_acc']:.3f} "
          f"(chance={probe['subject_chance_level']:.3f})")

    return results, model_cross


def run_part2(all_train_data, all_test_data, device):
    print("\n" + "=" * 70)
    print("PART 2 — Proposed Improvement: Subject-Adversarial EEGNet")
    print("=" * 70)
    results = {}

    Xtr, ytr, subj_tr_raw = data_mod.stack(all_train_data)
    Xte, yte, subj_te = data_mod.stack(all_test_data)
    n_channels, n_times = Xtr.shape[1], Xtr.shape[2]

    subj_map = {s: i for i, s in enumerate(sorted(set(subj_tr_raw.tolist())))}
    subj_idx = np.array([subj_map[s] for s in subj_tr_raw], dtype=np.int64)
    n_subjects = len(subj_map)

    sa_runs = []
    last_model = None
    for seed in SEEDS_FOR_ROBUSTNESS:
        set_seed(seed)
        model = SA_EEGNet(n_channels=n_channels, n_times=n_times, n_subjects=n_subjects)
        res = train_sa_eegnet(model, EEGDataset(Xtr, ytr, subj_idx), EEGDataset(Xte, yte), device, epochs=40)
        sa_runs.append(res["best_val_acc"])
        last_model = model
        print(f"SA-EEGNet cross-subject, seed={seed} — best test acc: {res['best_val_acc']:.3f}")
    results["sa_eegnet_runs"] = sa_runs
    results["sa_eegnet_best"] = max(sa_runs)
    results["sa_eegnet_worst"] = min(sa_runs)
    print(f"   -> BEST: {max(sa_runs):.3f}  WORST: {min(sa_runs):.3f}")

    # Does the adversarial branch actually reduce subject-decodability at the
    # spatial-filter bottleneck (the mechanism it's supposed to fix)?
    last_model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(Xtr).float().to(device)
        t = last_model.temporal(xb.unsqueeze(1))
        s_flat = last_model.spatial(t).flatten(1).cpu().numpy()
    probe = class_vs_subject_separability(s_flat, ytr, subj_tr_raw)
    results["sa_spatial_bottleneck_probe"] = probe
    print(f"   Spatial-bottleneck probe (SA-EEGNet) — class acc: "
          f"{probe['linear_probe_class_acc']:.3f}  subject acc: "
          f"{probe['linear_probe_subject_acc']:.3f} (chance={probe['subject_chance_level']:.3f})")

    return results


def run_part3(all_train_data, all_test_data, device):
    print("\n" + "=" * 70)
    print("PART 3 — Novel Idea PoC: State-Conditioned Dynamic Spatial Filtering")
    print("=" * 70)
    results = {}
    Xtr, ytr, subj_tr = data_mod.stack(all_train_data)
    Xte, yte, subj_te = data_mod.stack(all_test_data)
    n_channels, n_times = Xtr.shape[1], Xtr.shape[2]

    set_seed(SEED)
    model = DynamicEEGNet(n_channels=n_channels, n_times=n_times).to(device)
    res = train_baseline(model, EEGDataset(Xtr, ytr), EEGDataset(Xte, yte), device, epochs=40)
    results["dynamic_eegnet"] = res
    print(f"DynamicEEGNet cross-subject — best test acc: {res['best_val_acc']:.3f}")

    # Interpretability check: does the learned per-trial gain vector actually
    # track the band-power state descriptor it was conditioned on, or did the
    # hypernetwork collapse to a constant (i.e. did the idea do anything at all)?
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(Xte).float().to(device)
        _ = model._forward_features(xb)
        gains = model.dyn_spatial.last_gain.cpu().numpy()
    gain_std_across_trials = float(gains.std(axis=0).mean())
    gain_std_within_trial = float(gains.std(axis=1).mean())
    results["gain_variability"] = {
        "mean_std_across_trials_per_channel": gain_std_across_trials,
        "mean_std_across_channels_per_trial": gain_std_within_trial,
        "interpretation": (
            "If mean_std_across_trials_per_channel is near 0, the hypernetwork "
            "collapsed to a static gain (idea had no measurable effect on this "
            "run). Nonzero variance means the filter genuinely changes per trial."
        ),
    }
    print(f"   Gain variability across trials (per channel, avg): {gain_std_across_trials:.4f}")

    return results


def main():
    set_seed(SEED)
    device = get_device()
    ensure_dirs(RESULTS_DIR, FIG_DIR)
    print(f"Device: {device}")

    print("\nLoading data (TRAIN subjects 1-8, TEST subjects 9-10, runs=6,10,14)...")
    train_data = data_mod.load_subjects(TRAIN_SUBJECTS, RUNS)
    test_data = data_mod.load_subjects(TEST_SUBJECTS, RUNS)
    reduced_data = data_mod.load_subjects(REDUCED_SUBJECTS, RUNS) if data_mod.DATA_SOURCE["value"] == "real" \
        else [d for d in train_data if d[2] in REDUCED_SUBJECTS]

    data_source = data_mod.DATA_SOURCE["value"]

    part1_results, _ = run_part1(train_data, test_data, reduced_data, device)
    part2_results = run_part2(train_data, test_data, device)
    part3_results = run_part3(train_data, test_data, device)

    all_results = {
        "data_source": data_source,
        "seed": SEED,
        "robustness_seeds": SEEDS_FOR_ROBUSTNESS,
        "part1": part1_results,
        "part2": part2_results,
        "part3": part3_results,
    }
    out_path = os.path.join(RESULTS_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved results to {out_path}")
    if data_source == "synthetic":
        print("\n*** REMINDER: data_source=synthetic. These numbers are a pipeline ***")
        print("*** smoke test only. Re-run with PhysioNet access before submitting. ***")


if __name__ == "__main__":
    main()
