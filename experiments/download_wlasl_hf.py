"""Download WLASL video files from the Hugging Face mirror (Voxel51/WLASL), 8 at a time.

    python experiments/download_wlasl_hf.py experiments/hf_wlasl100_files.json datasets/WLASL-100/videos

The JSON maps each WLASL video_id to its path in the mirror. Videos are research-use only (C-UDA).
"""
import json, os, shutil, sys
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download
m = json.load(open(sys.argv[1])); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
def get(item):
    vid, path = item
    dst = os.path.join(out, vid + ".mp4")
    if os.path.exists(dst): return 0
    for attempt in range(3):
        try:
            src = hf_hub_download("Voxel51/WLASL", path, repo_type="dataset")
            shutil.copyfile(src, dst); return 0
        except Exception as e:
            err = e
    print("FAILED", vid, err, flush=True); return 1
with ThreadPoolExecutor(8) as ex: fails = sum(ex.map(get, m.items()))
print("done", len(os.listdir(out)), "files, failures", fails)
