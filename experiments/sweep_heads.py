"""Experiment helper: try several learning rates / dropout / weight-decay values on cached features and
print the best VALIDATION accuracy for each (the test set is never used here).

    python experiments/sweep_heads.py <features.pt>
"""
import sys, os, json, itertools, torch, numpy as np
sys.path.insert(0, os.getcwd())
from torch import nn
from ml_models.factory import build_model
d = torch.load(sys.argv[1]); dev = torch.device("cuda")
labels = json.load(open("datasets/WLASL-100/label_map.json")); y = torch.tensor([labels[l] for l in d["label"]])
split = np.array(d["split"]); tr, va, te = [torch.from_numpy(np.where(split == s)[0]) for s in ("train", "val", "test")]
def run(model_name, lr, dropout, wd, hidden=256, layers=2, epochs=120, seed=42, nframes=16, verbose=False):
    torch.manual_seed(seed); np.random.seed(seed)
    feats = (d["resnet"] if model_name == "cnn_lstm" else d["vit"]).float()
    if nframes != feats.shape[2]: feats = feats[:, :, torch.linspace(0, feats.shape[2] - 1, nframes).round().long()]
    feats = feats.to(dev); yy = y.to(dev)
    cfg = dict(dropout=dropout, pretrained=False, freeze_backbone=True, lstm_hidden_size=hidden, lstm_layers=layers, temporal_heads=8, temporal_layers=layers)
    m = build_model(model_name, 100, cfg).to(dev)
    params = [p for n, p in m.named_parameters() if not n.startswith("frame_encoder")]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1); V = feats.shape[0]
    best = (-1, 0, None)
    for ep in range(epochs):
        m.train(); perm = tr[torch.randperm(len(tr))]
        for b in perm.split(32):
            b = b.to(dev); views = torch.randint(0, V, (len(b),), device=dev)
            loss = lossf(m.forward_features(feats[views, b]), yy[b]); opt.zero_grad(); loss.backward(); opt.step()
        sched.step(); m.eval()
        with torch.no_grad():
            acc = lambda idx: (m.forward_features(feats[0, idx.to(dev)]).argmax(1) == yy[idx.to(dev)]).float().mean().item() * 100
            va_acc = acc(va)
            if va_acc > best[0]: best = (va_acc, ep + 1, None)
        if verbose and ep % 10 == 9: print(ep + 1, round(va_acc, 1))
    return best
if __name__ == "__main__":
    for model_name in ("cnn_lstm", "vit_temporal"):
        for lr, dropout, wd in itertools.product((1e-3, 3e-4), (0.3, 0.5), (1e-4, 1e-2)):
            va_acc, ep, te_acc = run(model_name, lr, dropout, wd)
            print(f"{model_name} lr={lr} dropout={dropout} wd={wd}: best val {va_acc:.1f}% @ep{ep}", flush=True)
