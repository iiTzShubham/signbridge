import sys, os, numpy as np, pandas as pd, torch, cv2, time
sys.path.insert(0, os.getcwd())
from torch.utils.data import Dataset, DataLoader
from ml_models.cnn_lstm import CNNLSTM
from ml_models.vit import VideoVisionTransformer
T, VIEWS, OUT = 16, int(sys.argv[2]), sys.argv[1]
CROP = sys.argv[3] if len(sys.argv) > 3 else "none"
MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1); STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
m = pd.read_csv("datasets/WLASL-100/manifest.csv")
class D(Dataset):
    def __len__(self): return len(m)
    def __getitem__(self, i):
        r = m.iloc[i]; cap = cv2.VideoCapture(os.path.join("datasets/WLASL-100", r.video_path)); fr = []
        while True:
            ok, f = cap.read()
            if not ok: break
            fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
        h0, w0 = fr[0].shape[:2]
        if CROP == "center" and w0 != h0:  # square crop centred on the signer
            side = min(h0, w0); x = (w0 - side) // 2; y = (h0 - side) // 2
            fr = [f[y:y + side, x:x + side] for f in fr]
        n = len(fr); h, w = fr[0].shape[:2]
        rng = np.random.default_rng(1000 + i); views = []
        for v in range(VIEWS):
            if v == 0: s, e, sc, cx, cy, flip = 0, n - 1, 1.0, .5, .5, False
            else:
                L = n - 1; s = int(rng.uniform(0, .15) * L); e = L - int(rng.uniform(0, .15) * L)
                sc = rng.uniform(.8, 1.0); cx, cy = rng.uniform(.4, .6), rng.uniform(.4, .6); flip = rng.random() < .5
            idx = np.linspace(s, max(s, e), T).round().astype(int)
            cw, ch = int(w * sc), int(h * sc); x0 = int(np.clip(cx * w - cw / 2, 0, w - cw)); y0 = int(np.clip(cy * h - ch / 2, 0, h - ch))
            out = []
            for k in idx:
                img = cv2.resize(fr[k][y0:y0 + ch, x0:x0 + cw], (224, 224), interpolation=cv2.INTER_AREA)
                out.append(img[:, ::-1] if flip else img)
            views.append(np.stack(out))
        return torch.from_numpy(np.stack(views).copy()), i
if __name__ == "__main__":
    dev = torch.device("cuda")
    res = CNNLSTM(100).frame_encoder.to(dev).eval(); vit = VideoVisionTransformer(100).frame_encoder.to(dev).eval()
    R = torch.zeros(VIEWS, len(m), T, 2048, dtype=torch.float16); V = torch.zeros(VIEWS, len(m), T, 768, dtype=torch.float16)
    t = time.time()
    with torch.inference_mode():
        for x, idx in DataLoader(D(), batch_size=2, num_workers=6):
            b = x.shape[0]; x = x.to(dev).flatten(0, 2).permute(0, 3, 1, 2).float().div(255)
            x = (x - MEAN.to(dev)) / STD.to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                fr_ = torch.cat([res(c).flatten(1) for c in x.split(128)]); fv = torch.cat([vit(c) for c in x.split(128)])
            R[:, idx] = fr_.view(b, VIEWS, T, -1).transpose(0, 1).half().cpu(); V[:, idx] = fv.view(b, VIEWS, T, -1).transpose(0, 1).half().cpu()
            if idx[0] % 100 == 0: print(int(idx[0]), round(time.time() - t), flush=True)
    torch.save({"resnet": R, "vit": V, "video_id": m.video_id.tolist(), "split": m.split.tolist(), "label": m.label.tolist()}, OUT)
    print("saved", OUT, round(time.time() - t), "s")
