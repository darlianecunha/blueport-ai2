"""Cross-validate and train the linear probe on CLIP embeddings produced by extract_features.py.

    python train_probe_cv.py            # 5-fold CV report, then final fit on all data

Outputs: eval_cv.json (report + confusion matrix), probe.npz (numpy weights used by the
Hugging Face Space) and blueport_linear_v2.pt (PyTorch state dict for waste_vision.py).
"""
import json
import numpy as np, torch
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score

X = np.load("feats.npy"); y = np.load("labels.npy"); classes = json.load(open("index.json"))["classes"]
X = X / np.linalg.norm(X, axis=1, keepdims=True)


def make():
    return LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")


pred = cross_val_predict(make(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42))
rep = classification_report(y, pred, target_names=classes, digits=3, output_dict=True)
cm = confusion_matrix(y, pred)
print(classification_report(y, pred, target_names=classes, digits=3))
print("balanced accuracy", round(balanced_accuracy_score(y, pred), 4))
print(cm)
json.dump({"report": rep, "cm": cm.tolist(), "cv": "5-fold stratified, seed 42"}, open("eval_cv.json", "w"), indent=1)

clf = make().fit(X, y)
np.savez("probe.npz", W=clf.coef_.astype(np.float32), b=clf.intercept_.astype(np.float32))
sd = OrderedDict([("weight", torch.tensor(clf.coef_, dtype=torch.float32)), ("bias", torch.tensor(clf.intercept_, dtype=torch.float32))])
torch.save({"state_dict": sd, "class_names": classes,
            "note": "logistic regression on L2-normalised CLIP ViT-B/32 image embeddings; normalise features before applying",
            "cv_accuracy_5fold": round(float(rep["accuracy"]), 4)}, "blueport_linear_v2.pt")
print("saved probe.npz and blueport_linear_v2.pt")
