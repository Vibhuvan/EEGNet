"""
Part 2 — Proposed improvement: Subject-Adversarial EEGNet (SA-EEGNet).

ROOT CAUSE (from Part 1 analysis, see ANALYSIS.md):
EEGNet's depthwise spatial filters (Conv2d(F1, F1*D, (n_channels,1))) are
a *global, fixed* linear combination over electrodes, learned once on the
training subjects. Because electrode-to-cortical-source geometry differs
across heads (skull thickness, gyral folding, cap placement offset), a
spatial filter that isolates motor cortex signal for training subjects'
head geometry will project a different — and partially wrong — mixture
of sources for a new subject. The filters are not necessarily learning
"task-relevant neural signatures" in a subject-general sense; they are
free to shortcut onto whatever linear combination separates classes in
the training subjects' specific electrode geometry, including
subject-identifying information, because nothing in the loss discourages
that.

WHY A SYMPTOM-LEVEL FIX WOULD BE WRONG:
More dropout, more L2, or simple data augmentation (symptom-level fixes)
shrink the model's capacity to overfit *anything* — including the
task-relevant signal — rather than specifically remove the subject-ID
component from what the spatial filters encode. They don't touch the
mechanism.

THE FIX (root-cause level):
Attach a small subject-identity classifier that reads off the SAME
spatial-filter output (post block-1 features) that feeds the task
classifier, and connect it to the spatial filters through a Gradient
Reversal Layer (GRL, Ganin & Lempitsky 2015). During backprop, the GRL
negates the gradient from the subject-classification loss before it
reaches the spatial filters. This means the spatial filters are
explicitly optimized to:
  (a) retain information that predicts left/right fist (task loss, normal sign)
  (b) actively DISCARD information that predicts subject identity (adversarial loss, reversed sign)
This is the intervention this failure analysis actually implies: force
the spatial filter bank to stop encoding subject identity as a
side-channel, rather than hoping added regularization removes it as a
side effect.

This is inspired by domain-adversarial training (DANN) but adapted: DANN
is normally used with a *domain label* (e.g. dataset A vs B); here the
adversarial target is *subject ID as a multi-class nuisance variable*
applied at the spatial-filter bottleneck specifically (not at the final
feature layer, where DANN is usually applied) because that is the exact
layer diagnosed as the failure point in Part 1. Applying it there, rather
than at the penultimate layer, is the deliberate design choice that
differs from a direct DANN transplant.
"""
import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x, lambda_=1.0):
    return GradientReversal.apply(x, lambda_)


class SA_EEGNet(nn.Module):
    def __init__(self, n_channels=64, n_times=256, n_classes=2, n_subjects=8,
                 F1=8, D=2, F2=16, kernel_length=64, dropout=0.5, grl_lambda=0.3):
        super().__init__()
        self.grl_lambda = grl_lambda

        self.temporal = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            spatial_feat_dim = self._spatial_features(dummy).shape[1]
            full_feat_dim = self._forward_features(dummy).shape[1]

        self.classifier = nn.Linear(full_feat_dim, n_classes)
        # subject discriminator reads off the SPATIAL-FILTER output directly
        # (the diagnosed bottleneck), not the final task feature
        self.subject_head = nn.Sequential(
            nn.Linear(spatial_feat_dim, 32),
            nn.ELU(),
            nn.Linear(32, n_subjects),
        )

    def _spatial_features(self, x):
        x = self.temporal(x)
        x = self.spatial(x)
        return x.flatten(1)

    def _forward_features(self, x):
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.separable(x)
        return x.flatten(1)

    def forward(self, x, return_all=False):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        t = self.temporal(x)
        s = self.spatial(t)
        s_flat = s.flatten(1)  # this is the layer we attach the GRL to

        sep = self.separable(s)
        task_feat = sep.flatten(1)
        task_logits = self.classifier(task_feat)

        subj_logits = self.subject_head(grad_reverse(s_flat, self.grl_lambda))

        if return_all:
            return task_logits, subj_logits, task_feat
        return task_logits
