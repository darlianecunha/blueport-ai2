"""Extract L2-normalised CLIP ViT-B/32 image embeddings for every image in dataset/<class>/.

Writes feats.npy (N x 512), labels.npy (N) and index.json (class order and file paths).
Uses the Hugging Face implementation of CLIP so that the same code runs in the Space.

    python extract_features.py --dataset dataset --out .
"""
import argparse, json, os, time
import numpy as np, torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_ID = "openai/clip-vit-base-patch32"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--out", default=".")
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()

    classes = sorted(d for d in os.listdir(a.dataset) if os.path.isdir(os.path.join(a.dataset, d)))
    paths, labels = [], []
    for i, c in enumerate(classes):
        for f in sorted(os.listdir(os.path.join(a.dataset, c))):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(a.dataset, c, f)); labels.append(i)
    print(f"{len(paths)} images, classes: {classes}")

    model = CLIPModel.from_pretrained(MODEL_ID).eval()
    proc = CLIPProcessor.from_pretrained(MODEL_ID)
    feats = np.zeros((len(paths), 512), dtype=np.float32)
    t0 = time.time()
    for s in range(0, len(paths), a.batch):
        imgs = []
        for p in paths[s:s + a.batch]:
            try:
                imgs.append(Image.open(p).convert("RGB"))
            except Exception:
                imgs.append(Image.new("RGB", (224, 224)))
        with torch.no_grad():
            out = model.get_image_features(**proc(images=imgs, return_tensors="pt"))
        e = out.image_embeds if hasattr(out, "image_embeds") else (out.pooler_output if hasattr(out, "pooler_output") else out)
        feats[s:s + len(imgs)] = e.numpy()
        if (s // a.batch) % 20 == 0:
            print(f"{s + len(imgs)}/{len(paths)}  {time.time() - t0:.0f}s")
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    np.save(os.path.join(a.out, "feats.npy"), feats)
    np.save(os.path.join(a.out, "labels.npy"), np.array(labels))
    json.dump({"classes": classes, "paths": paths, "labels": labels}, open(os.path.join(a.out, "index.json"), "w"))


if __name__ == "__main__":
    main()
