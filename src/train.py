"""Generic training/eval loop shared by baseline, SA-EEGNet, and DynamicEEGNet."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class EEGDataset(Dataset):
    def __init__(self, X, y, subj=None):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.subj = torch.from_numpy(subj).long() if subj is not None else None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.subj is not None:
            return self.X[idx], self.y[idx], self.subj[idx]
        return self.X[idx], self.y[idx]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        logits = model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += len(y)
    return correct / max(total, 1)


def train_baseline(model, train_ds, val_ds, device, epochs=60, lr=1e-3, batch_size=32, verbose=False):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    model.to(device)

    best_val = 0.0
    history = []
    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
        val_acc = evaluate(model, val_loader, device)
        best_val = max(best_val, val_acc)
        history.append(val_acc)
        if verbose and (ep + 1) % 10 == 0:
            print(f"  epoch {ep+1}/{epochs}  val_acc={val_acc:.3f}")
    return {"history": history, "final_val_acc": history[-1], "best_val_acc": best_val}


def train_sa_eegnet(model, train_ds, val_ds, device, epochs=60, lr=1e-3, batch_size=32,
                     subj_loss_weight=1.0, verbose=False):
    """train_ds must yield (x, y, subj_idx) triples with subj_idx in [0, n_subjects)."""
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    task_crit = nn.CrossEntropyLoss()
    subj_crit = nn.CrossEntropyLoss()
    model.to(device)

    best_val = 0.0
    history = []
    for ep in range(epochs):
        model.train()
        for x, y, subj in train_loader:
            x, y, subj = x.to(device), y.to(device), subj.to(device)
            opt.zero_grad()
            task_logits, subj_logits, _ = model(x, return_all=True)
            loss = task_crit(task_logits, y) + subj_loss_weight * subj_crit(subj_logits, subj)
            loss.backward()
            opt.step()
        val_acc = evaluate(model, val_loader, device)
        best_val = max(best_val, val_acc)
        history.append(val_acc)
        if verbose and (ep + 1) % 10 == 0:
            print(f"  epoch {ep+1}/{epochs}  val_acc={val_acc:.3f}")
    return {"history": history, "final_val_acc": history[-1], "best_val_acc": best_val}


def extract_features(model, dataset, device, batch_size=64):
    """Run model in eval mode and collect penultimate-layer features + labels + subject ids."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    feats, labels, subjs = [], [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                x, y, s = batch
                subjs.append(s.numpy())
            else:
                x, y = batch
            x = x.to(device)
            if hasattr(model, "return_features"):
                model.return_features = True
                logits, f = model(x)
                model.return_features = False
            else:
                f = model._forward_features(x.unsqueeze(1) if x.dim() == 3 else x)
            feats.append(f.cpu().numpy())
            labels.append(y.numpy())
    feats = np.concatenate(feats)
    labels = np.concatenate(labels)
    subjs = np.concatenate(subjs) if subjs else None
    return feats, labels, subjs
