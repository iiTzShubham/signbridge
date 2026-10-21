"""Fine-tune backbone + temporal head end-to-end on WLASL-100 (same recipe for both models)."""
import argparse, json, os, sys, time
import cv2, numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.getcwd())
from ml_models.factory import build_model

MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)


class Clips(Dataset):
    def __init__(self, rows, labels, frames, train):
        self.rows, self.labels, self.T, self.train = rows.reset_index(drop=True), labels, frames, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        cap = cv2.VideoCapture(os.path.join("datasets/WLASL-100", r.video_path)); fr = []
        while True:
            ok, f = cap.read()
            if not ok: break
            fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
        h, w = fr[0].shape[:2]; side = min(h, w)
        n = len(fr) - 1
        if self.train:
            rng = np.random.default_rng()
            s, e = int(rng.uniform(0, .15) * n), n - int(rng.uniform(0, .15) * n)
            sc = rng.uniform(.8, 1.0); cw = int(side * sc)
            x0 = int(np.clip(rng.uniform(.4, .6) * w - cw / 2, 0, w - cw)); y0 = int(np.clip(rng.uniform(.4, .6) * h - cw / 2, 0, h - cw))
            flip = rng.random() < .5
        else:
            s, e, cw, flip = 0, n, side, False
            x0, y0 = (w - side) // 2, (h - side) // 2
        idx = np.linspace(s, max(s, e), self.T).round().astype(int)
        out = []
        for k in idx:
            img = cv2.resize(fr[k][y0:y0 + cw, x0:x0 + cw], (224, 224), interpolation=cv2.INTER_AREA)
            out.append(img[:, ::-1] if flip else img)
        x = torch.from_numpy(np.stack(out).copy()).permute(0, 3, 1, 2)
        return x, self.labels[r.label]


def prep(x, dev):
    x = x.to(dev, non_blocking=True).float().div_(255)
    return (x - MEAN.to(dev).unsqueeze(0)) / STD.to(dev).unsqueeze(0)


def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d): m.eval()


def evaluate(model, loader, dev):
    model.eval(); correct = n = 0; loss = 0.0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for x, y in loader:
            y = y.to(dev); logits = model(prep(x, dev))
            loss += nn.functional.cross_entropy(logits.float(), y, reduction="sum").item()
            correct += (logits.argmax(1) == y).sum().item(); n += len(y)
    return correct / n * 100, loss / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="cnn_lstm"); p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--frames", type=int, default=16); p.add_argument("--batch", type=int, default=4)
    p.add_argument("--accum", type=int, default=2); p.add_argument("--lr-backbone", type=float, default=3e-5)
    p.add_argument("--lr-head", type=float, default=3e-4); p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--out", default="experiments/ft")
    a = p.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); dev = torch.device("cuda")
    m = pd.read_csv("datasets/WLASL-100/manifest.csv"); labels = json.load(open("datasets/WLASL-100/label_map.json"))
    tr = DataLoader(Clips(m[m.split == "train"], labels, a.frames, True), batch_size=a.batch, shuffle=True, num_workers=6, persistent_workers=True, drop_last=True)
    va = DataLoader(Clips(m[m.split == "val"], labels, a.frames, False), batch_size=a.batch, num_workers=4, persistent_workers=True)
    cfg = dict(dropout=a.dropout, pretrained=True, freeze_backbone=False, lstm_hidden_size=256, lstm_layers=2, temporal_heads=8, temporal_layers=2)
    model = build_model(a.model, 100, cfg).to(dev)
    bb = [p_ for n_, p_ in model.named_parameters() if n_.startswith("frame_encoder")]
    head = [p_ for n_, p_ in model.named_parameters() if not n_.startswith("frame_encoder")]
    opt = torch.optim.AdamW([{"params": bb, "lr": a.lr_backbone}, {"params": head, "lr": a.lr_head}], weight_decay=0.05)
    steps = a.epochs * (len(tr) // a.accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr_backbone, a.lr_head], total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(); lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
    os.makedirs(a.out, exist_ok=True); ckpt = os.path.join(a.out, f"{a.model}_seed{a.seed}.pth"); best = -1
    for ep in range(1, a.epochs + 1):
        t = time.time(); model.train(); freeze_bn(model); tl = 0.0
        for i, (x, y) in enumerate(tr):
            with torch.autocast("cuda", dtype=torch.float16):
                loss = lossf(model(prep(x, dev)), y.to(dev)) / a.accum
            scaler.scale(loss).backward(); tl += loss.item() * a.accum
            if (i + 1) % a.accum == 0:
                scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
        acc, vl = evaluate(model, va, dev)
        if acc > best:
            best = acc; torch.save({"state_dict": model.state_dict(), "best_epoch": ep, "val_top1": acc}, ckpt)
        print(f"{a.model} ep{ep} train_loss={tl / len(tr):.3f} val_loss={vl:.3f} val_top1={acc:.1f}% best={best:.1f}% {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
