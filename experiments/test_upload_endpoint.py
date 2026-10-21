"""Send video clips through the running web app's /recognize/ upload form and record the predictions.

Usage: python experiments/test_upload_endpoint.py <list.csv with columns path,label> <out.csv> [base_url]
"""
import csv
import re
import sys
import time

import requests

src, out = sys.argv[1], sys.argv[2]
base = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8000"
session = requests.Session()
page = session.get(f"{base}/recognize/")
token = session.cookies.get("csrftoken") or re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', page.text).group(1)
rows = list(csv.DictReader(open(src, encoding="utf-8")))
results = []
for i, row in enumerate(rows, 1):
    for model in ("vit_temporal", "cnn_lstm"):
        started = time.perf_counter()
        with open(row["path"], "rb") as handle:
            response = session.post(f"{base}/recognize/", data={"model": model, "csrfmiddlewaretoken": token},
                                    files={"video": (row["path"].rsplit("/", 1)[-1], handle, "video/mp4")},
                                    headers={"Accept": "application/json", "Referer": f"{base}/recognize/"})
        payload = response.json()
        result = payload.get("result") or {}
        top = [p["label"] for p in result.get("top_predictions", [])]
        results.append({
            "path": row["path"], "label": row["label"], "model": model, "predicted": result.get("label", ""),
            "confidence_pct": round(result.get("confidence", 0), 1), "top3": "|".join(top),
            "top1_correct": int(result.get("label") == row["label"]), "top3_correct": int(row["label"] in top),
            "model_ms": round(result.get("latency_ms", 0), 1), "request_ms": round((time.perf_counter() - started) * 1000, 1),
            "error": payload.get("error", ""),
        })
    if i % 50 == 0:
        print(i, "clips", flush=True)
with open(out, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(results[0]))
    writer.writeheader()
    writer.writerows(results)
print("wrote", out)
