# -*- coding: utf-8 -*-
"""
train_linear_probe.py — Treinamento leve (linear probe) para o BluePort
Usa o CLIP apenas como extrator de features e aprende uma camada linear.
Gera o arquivo 'blueport_linear.pt' para uso no waste_vision.py.
"""

import os
import torch
import clip
from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# Configurações
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-3
DATASET_DIR = "dataset"
MODEL_OUT = "blueport_linear.pt"


# ---------------------------------------------------------
# Dataset simples: extrai embeddings CLIP e rótulo numérico
# ---------------------------------------------------------
class WasteDataset(Dataset):
    def __init__(self, folder_path, preprocess):
        self.samples = []
        self.labels = []
        self.class_names = sorted(os.listdir(folder_path))
        for idx, cls in enumerate(self.class_names):
            cls_path = os.path.join(folder_path, cls)
            if not os.path.isdir(cls_path):
                continue
            for fname in os.listdir(cls_path):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append(os.path.join(cls_path, fname))
                    self.labels.append(idx)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.preprocess(image)
        label = self.labels[idx]
        return image, label


# ---------------------------------------------------------
# Função de treino
# ---------------------------------------------------------
def train_probe():
    print("Carregando CLIP (ViT-B/32)...")
    model, preprocess = clip.load("ViT-B/32", device=DEVICE, jit=False)
    model.eval()

    # Dataset
    ds = WasteDataset(DATASET_DIR, preprocess)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    num_classes = len(ds.class_names)
    print(f"🔹 {len(ds)} imagens, {num_classes} classes: {', '.join(ds.class_names)}")

    # Camada linear para cima das features CLIP
    probe = nn.Linear(model.visual.output_dim, num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(probe.parameters(), lr=LR)

    # Treinamento
    for epoch in range(EPOCHS):
        total_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(loader, desc=f"Época {epoch+1}/{EPOCHS}"):
            with torch.no_grad():
                feats = model.encode_image(images.to(DEVICE))
                feats = feats / feats.norm(dim=-1, keepdim=True)
            outputs = probe(feats)
            loss = criterion(outputs, labels.to(DEVICE))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds.cpu() == labels).sum().item()
            total += labels.size(0)

        acc = 100 * correct / total
        print(f"🟩 Época {epoch+1}: Loss {total_loss/len(loader):.4f}, Acurácia {acc:.1f}%")

    # Salva modelo e nomes das classes
    torch.save({
        "state_dict": probe.state_dict(),
        "class_names": ds.class_names,
    }, MODEL_OUT)
    print(f"✅ Modelo salvo em {MODEL_OUT}")


if __name__ == "__main__":
    train_probe()
