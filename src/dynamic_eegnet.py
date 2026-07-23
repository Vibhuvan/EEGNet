"""
Part 3 — "One thing nobody has tried": State-Conditioned Dynamic Spatial
Filtering (SC-DSF).

THE GAP:
Every EEGNet-family model (and most EEG-CNN literature we're aware of)
uses spatial filters that are STATIC per trial: one learned weight per
electrode, applied identically regardless of the subject's ongoing
oscillatory state at trial onset. But the ERD/ERS literature (Pfurtscheller
& Lopes da Silva, 1999) establishes that motor-imagery-related
desynchronization is a *relative* change from a subject's pre-trial
"idle" mu/beta baseline — and that baseline power varies substantially
trial-to-trial (arousal, fatigue, attention) and subject-to-subject
(individual alpha/mu power varies by an order of magnitude across
people). A fixed spatial filter implicitly assumes a fixed
signal-to-baseline regime; it cannot rescale itself for a trial where
the subject's baseline mu power is unusually high or low, even though
the optimal spatial projection to isolate ERD arguably depends on that
baseline.

THE IDEA:
Make the spatial filter a FUNCTION of a lightweight, per-trial state
descriptor (band power per channel in mu/beta, computed from the
opening ~250ms of the trial) rather than a fixed parameter. A small
hypernetwork maps this state vector to a per-channel gain vector that
multiplicatively modulates a base (still-learned) spatial filter before
the depthwise convolution is applied. This is "dynamic convolution"
(Chen et al., 2020, computer vision) adapted to EEG and conditioned
specifically on a neuroscience-motivated state signal, rather than on
generic input statistics — which is the part we believe is new for EEG
spatial filtering.

WHY THIS IS GROUNDED / SPECIFIC / DEFENSIBLE:
- Grounded: directly targets a documented neurophysiological
  confound (baseline-dependent ERD magnitude) rather than a generic
  "more flexible model" argument.
- Specific: a concrete architectural change (state descriptor ->
  hypernetwork -> multiplicative per-channel gain on a named layer),
  not "add more layers."
- Defensible: if baseline state truly modulates the optimal spatial
  projection, a model that can rescale its filters per-trial should
  need less capacity elsewhere to compensate for that variability,
  and its embeddings should show tighter within-class clustering
  independent of baseline-power outliers. That's a testable, falsifiable
  claim, which is exactly what we test in Part 3 of ANALYSIS.md.

WHAT THIS POC DOES / DOES NOT CLAIM:
This is a minimal proof-of-concept, not a validated method. It exists
to show the idea translates into working code and produces an
interpretable result (does the learned gain vector actually correlate
with the state descriptor, and does performance/embedding structure
move in the predicted direction), not to claim a state-of-the-art
result on 10 subjects of data.
"""
import torch
import torch.nn as nn


def band_power_state(x, fs=128, band=(8, 30)):
    """
    Cheap per-channel band-power descriptor computed via FFT magnitude,
    averaged over the requested band. x: (batch, n_channels, n_times).
    Returns (batch, n_channels), log-scaled and z-normalized per batch
    so the hypernetwork sees a well-conditioned input.
    """
    n_times = x.shape[-1]
    freqs = torch.fft.rfftfreq(n_times, d=1.0 / fs).to(x.device)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    spec = torch.fft.rfft(x, dim=-1)
    power = (spec.abs() ** 2)[..., mask].mean(dim=-1)  # (batch, n_channels)
    power = torch.log(power + 1e-6)
    power = (power - power.mean(dim=1, keepdim=True)) / (power.std(dim=1, keepdim=True) + 1e-6)
    return power


class DynamicSpatialFilter(nn.Module):
    """
    Replaces EEGNet's static depthwise spatial conv with a state-modulated
    version: base learned filter (F1*D, n_channels), elementwise-scaled
    per trial by a gain vector g(state) in (0, 2), produced by a tiny
    hypernetwork from the per-trial band-power descriptor.
    """
    def __init__(self, n_channels, F1, D):
        super().__init__()
        self.n_channels = n_channels
        self.F1, self.D = F1, D
        self.base_filter = nn.Parameter(torch.randn(F1 * D, n_channels) * 0.1)
        self.hyper = nn.Sequential(
            nn.Linear(n_channels, 32),
            nn.ELU(),
            nn.Linear(32, n_channels),
            nn.Sigmoid(),  # gain in (0,1), rescaled to (0,2) below
        )
        self.bn = nn.BatchNorm2d(F1 * D)

    def forward(self, temporal_feats, raw_x):
        # temporal_feats: (batch, F1, n_channels, n_times) post block-1 temporal conv
        # raw_x: (batch, n_channels, n_times) original signal, used only to derive state
        state = band_power_state(raw_x)              # (batch, n_channels)
        gain = self.hyper(state) * 2.0                # (batch, n_channels), in (0,2)

        b = temporal_feats.shape[0]
        # modulate the base filter per-trial: (batch, F1*D, n_channels)
        filt = self.base_filter.unsqueeze(0) * gain.unsqueeze(1)  # broadcast over F1*D
        # apply as a per-trial "depthwise-ish" linear combination over channels.
        # temporal_feats: (b, F1, C, T) -> repeat each F1 map D times to match F1*D outputs
        tf = temporal_feats.repeat_interleave(self.D, dim=1)      # (b, F1*D, C, T)
        # weighted sum over channel dim C using the per-trial filter
        out = torch.einsum('bfct,bfc->bft', tf, filt)             # (b, F1*D, T)
        out = out.unsqueeze(2)                                    # (b, F1*D, 1, T) to keep conv2d-style shape
        out = self.bn(out)
        self.last_gain = gain.detach()
        return out


class DynamicEEGNet(nn.Module):
    def __init__(self, n_channels=64, n_times=256, n_classes=2,
                 F1=8, D=2, F2=16, kernel_length=64, dropout=0.5):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.dyn_spatial = DynamicSpatialFilter(n_channels, F1, D)
        self.post_spatial = nn.Sequential(
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
            dummy = torch.zeros(1, n_channels, n_times)
            feat_dim = self._forward_features(dummy).shape[1]
        self.classifier = nn.Linear(feat_dim, n_classes)

    def _forward_features(self, x):
        raw = x
        xin = x.unsqueeze(1) if x.dim() == 3 else x
        t = self.temporal(xin)                 # (b, F1, C, T)
        s = self.dyn_spatial(t, raw.squeeze(1) if raw.dim() == 4 else raw)  # (b, F1*D, 1, T)
        s = self.post_spatial(s)
        sep = self.separable(s)
        return sep.flatten(1)

    def forward(self, x):
        feats = self._forward_features(x)
        return self.classifier(feats)
