"""t-SNE / UMAP visualization of EEGNet's learned (penultimate-layer) representations."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


def plot_embedding(feats, labels, subj_ids, title, out_path, method="tsne"):
    n = len(feats)
    perplexity = min(30, max(5, n // 4))
    if method == "tsne":
        emb = TSNE(n_components=2, random_state=42, perplexity=perplexity, init="pca").fit_transform(feats)
    elif method == "umap" and HAS_UMAP:
        emb = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, n - 1)).fit_transform(feats)
    else:
        emb = TSNE(n_components=2, random_state=42, perplexity=perplexity, init="pca").fit_transform(feats)
        method = "tsne (umap unavailable)"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # colored by class
    for cls in np.unique(labels):
        m = labels == cls
        axes[0].scatter(emb[m, 0], emb[m, 1], s=12, alpha=0.7,
                         label=f"class {cls}")
    axes[0].set_title(f"{title}\ncolored by class")
    axes[0].legend(fontsize=8)

    # colored by subject
    for subj in np.unique(subj_ids):
        m = subj_ids == subj
        axes[1].scatter(emb[m, 0], emb[m, 1], s=12, alpha=0.7, label=f"S{subj}")
    axes[1].set_title(f"{title}\ncolored by subject")
    axes[1].legend(fontsize=7, ncol=2)

    fig.suptitle(f"method={method}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def class_vs_subject_separability(feats, labels, subj_ids):
    """
    Cheap quantitative companion to the embedding plot: how well does a
    trivial linear probe separate CLASS vs how well does it separate
    SUBJECT, from the same features? If subject is easier to decode than
    class, that's direct evidence the representation is subject-coded
    rather than task-coded.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    class_acc = cross_val_score(
        LogisticRegression(max_iter=2000), feats, labels, cv=5).mean()
    subj_acc = cross_val_score(
        LogisticRegression(max_iter=2000), feats, subj_ids, cv=5).mean()
    chance_subj = 1.0 / len(np.unique(subj_ids))
    return {
        "linear_probe_class_acc": float(class_acc),
        "linear_probe_subject_acc": float(subj_acc),
        "subject_chance_level": float(chance_subj),
    }
