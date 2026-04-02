# -*- coding: utf-8 -*-
"""
Núcleo BluePort AI (visão):
- Contador (count.txt) e log (log_results.csv)
- CLIP zero-shot com prompts otimizados
- Pré-processamento reforçado (RGB + 224x224)
- Softmax com temperatura + limiar de rejeição
- Carregamento automático do modelo adaptado (linear probe): blueport_linear.pt
- Utilitários para o bot: current_mode(), refresh_model()
"""

import os
import io
import csv
import json
import pathlib
import datetime as dt
from typing import List, Dict, Tuple, Optional, Union

import torch
from torch import nn
from PIL import Image

try:
    import clip  # pip install git+https://github.com/openai/CLIP.git
except Exception as e:
    raise RuntimeError(
        "CLIP não está instalado. Use: pip install git+https://github.com/openai/CLIP.git"
    ) from e

# =========================
# Configurações
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Durante testes, pode usar 0.10; depois suba p/ 0.35–0.5
MIN_CONFIDENCE = 0.10
SOFTMAX_TEMPERATURE = 0.10

COUNT_FILE = "count.txt"
LOG_FILE = "log_results.csv"

# caminho absoluto do modelo adaptado (ao lado deste arquivo)
ADAPTED_MODEL_PATH = str(pathlib.Path(__file__).with_name("blueport_linear.pt"))

CLASSES = ["plástico", "papel", "metal", "vidro", "orgânico", "eletrônico", "madeira", "têxtil", "misto"]

CLASS_PROMPTS: Dict[str, List[str]] = {
    "plástico": [
        "photo of plastic waste bottle",
        "photo of used plastic packaging",
        "photo of discarded plastic item like bottle or container",
        "recyclable plastic waste",
        "plastic garbage ready for recycling",
    ],
    "papel": [
        "photo of paper or cardboard waste",
        "photo of paper waste and cardboard boxes in trash",
        "recyclable paper or cardboard garbage",
        "piles of paper sheets and boxes in a recycling bin",
    ],
    "metal": [
        "photo of metal cans in waste",
        "photo of aluminum beverage can for recycling",
        "metal scrap or metallic waste",
    ],
    "vidro": [
        "photo of glass bottle waste",
        "photo of broken glass or glass cup waste",
        "recyclable glass container",
    ],
    "orgânico": [
        "photo of food scraps or organic waste",
        "compostable organic garbage",
    ],
    "eletrônico": [
        "photo of discarded electronic device or cable",
        "photo of electronic waste",
    ],
    "madeira": [
        "photo of wood scrap or wooden waste",
        "wood pieces discarded",
    ],
    "têxtil": [
        "photo of discarded clothes or fabric waste",
        "textile waste for recycling",
    ],
    "misto": [
        "photo of mixed household trash with plastic and paper",
        "unsorted recyclable waste mix",
    ],
}

# =========================
# Contador e log
# =========================
def _init_count_file() -> None:
    if not os.path.exists(COUNT_FILE):
        with open(COUNT_FILE, "w") as f:
            f.write("0")

def _read_count() -> int:
    _init_count_file()
    try:
        with open(COUNT_FILE, "r") as f:
            return int(f.read().strip())
    except:
        return 0

def _write_count(value: int) -> None:
    with open(COUNT_FILE, "w") as f:
        f.write(str(value))

def get_count() -> int:
    return _read_count()

def _increment_count() -> int:
    count = _read_count() + 1
    _write_count(count)
    return count

def _init_log_file() -> None:
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp", "filename", "label", "confidence", "top3"])

def _log_result(filename: Optional[str], label: str, confidence: float, top3: List[Tuple[str, float]]) -> None:
    _init_log_file()
    ts = dt.datetime.now().isoformat(timespec="seconds")
    top3_str = json.dumps([{"label": l, "conf": round(c, 4)} for l, c in top3], ensure_ascii=False)
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([ts, filename or "", label, round(confidence, 4), top3_str])

def get_stats() -> Dict[str, Union[int, float]]:
    if not os.path.exists(LOG_FILE):
        return {"total": 0, "media_confianca": 0.0}
    total = 0
    soma = 0.0
    with open(LOG_FILE, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            total += 1
            try:
                soma += float(r.get("confidence", 0.0))
            except:
                pass
    return {"total": total, "media_confianca": (soma / total) if total else 0.0}

# =========================
# Modelo CLIP + Linear Probe
# =========================
_model = None
_preprocess = None
_text_features_per_class = None
_all_class_names = CLASSES[:]  # zero-shot

_linear_probe: Optional[nn.Linear] = None
_adapted_class_names: Optional[List[str]] = None

def _clean_class_names(names: List[str]) -> List[str]:
    # tira espaços e ignora entradas vazias; NÃO remove classes (mantém dimensões do modelo salvo)
    return [n.strip() if isinstance(n, str) else n for n in names]

def _load_model():
    global _model, _preprocess, _linear_probe, _adapted_class_names

    if _model is None or _preprocess is None:
        _model, _preprocess = clip.load("ViT-B/32", device=DEVICE, jit=False)
        _model.eval()

        if os.path.exists(ADAPTED_MODEL_PATH):
            ckpt = torch.load(ADAPTED_MODEL_PATH, map_location=DEVICE)
            class_names = ckpt.get("class_names")
            state_dict = ckpt.get("state_dict")
            if class_names and state_dict:
                class_names = _clean_class_names(class_names)
                probe = nn.Linear(_model.visual.output_dim, len(class_names)).to(DEVICE)
                probe.load_state_dict(state_dict)
                probe.eval()
                _linear_probe = probe
                _adapted_class_names = class_names
                print(f"🔹 Modelo adaptado encontrado: {len(class_names)} classes")
            else:
                print("⚠️ blueport_linear.pt inválido — usando zero-shot.")
        else:
            print("ℹ️ Nenhum modelo adaptado encontrado — usando zero-shot.")
    return _model, _preprocess

@torch.no_grad()
def _build_text_features() -> torch.Tensor:
    global _text_features_per_class
    if _text_features_per_class is not None:
        return _text_features_per_class
    model, _ = _load_model()
    embs = []
    for cls in _all_class_names:
        prompts = CLASS_PROMPTS.get(cls, [f"a photo of {cls} waste"])
        toks = clip.tokenize(prompts).to(DEVICE)
        tfeat = model.encode_text(toks)
        tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)
        embs.append(tfeat.mean(dim=0))
    feats = torch.stack(embs, dim=0)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    _text_features_per_class = feats
    return feats

def _softmax_with_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.softmax(logits / max(temperature, 1e-6), dim=-1)

# =========================
# Inferência
# =========================
@torch.no_grad()
def _predict_from_pil(img: Image.Image) -> Tuple[str, float, List[Tuple[str, float]]]:
    model, preprocess = _load_model()

    # reforço de pré-processamento
    img = img.convert("RGB").resize((224, 224))

    # CLIP preprocess
    image_input = preprocess(img).unsqueeze(0).to(DEVICE)
    image_features = model.encode_image(image_input)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # caminho 1: modelo adaptado
    if _linear_probe is not None and _adapted_class_names is not None:
        logits = _linear_probe(image_features)            # [1, K]
        probs = torch.softmax(logits, dim=-1).squeeze(0)  # [K]
        conf_vals = probs.tolist()
        idx_sorted = torch.argsort(probs, descending=True).tolist()
        top3 = [(_adapted_class_names[i], float(conf_vals[i])) for i in idx_sorted[:3]]
        best_idx = idx_sorted[0]
        best_label = _adapted_class_names[best_idx]
        best_conf = float(conf_vals[best_idx])
    else:
        # caminho 2: zero-shot
        text_features = _build_text_features()                # [C, dim]
        sims = (image_features @ text_features.T).squeeze(0)  # [C]
        probs = _softmax_with_temperature(sims, SOFTMAX_TEMPERATURE)
        conf_vals = probs.tolist()
        idx_sorted = torch.argsort(probs, descending=True).tolist()
        top3 = [(_all_class_names[i], float(conf_vals[i])) for i in idx_sorted[:3]]
        best_idx = idx_sorted[0]
        best_label = _all_class_names[best_idx]
        best_conf = float(conf_vals[best_idx])

    # limiar
    if best_conf < MIN_CONFIDENCE:
        return "desconhecido", best_conf, top3
    return best_label, best_conf, top3

def current_mode() -> str:
    """Retorna 'adapted' se usar linear probe; senão 'zeroshot'."""
    return "adapted" if (_linear_probe is not None and _adapted_class_names is not None) else "zeroshot"

def refresh_model():
    """Recarrega CLIP e (se existir) o linear probe do zero."""
    global _model, _preprocess, _text_features_per_class, _linear_probe, _adapted_class_names
    _model = None
    _preprocess = None
    _text_features_per_class = None
    _linear_probe = None
    _adapted_class_names = None
    _load_model()

# =========================
# APIs públicas
# =========================
def analyze(image: Union[Image.Image, bytes], filename: Optional[str] = None) -> Dict[str, Union[str, float, int, list]]:
    if isinstance(image, bytes):
        image = Image.open(io.BytesIO(image))
    if not isinstance(image, Image.Image):
        raise ValueError("analyze: 'image' deve ser PIL.Image ou bytes")
    image = image.convert("RGB")
    label, confidence, top3 = _predict_from_pil(image)
    total = _increment_count()
    _log_result(filename, label, confidence, top3)
    return {
        "label": label,
        "confidence": round(confidence, 4),
        "top3": [(l, round(c, 4)) for l, c in top3],
        "count_total": total,
    }

def predict(image: Image.Image) -> Tuple[str, float]:
    if not isinstance(image, Image.Image):
        raise ValueError("predict: 'image' deve ser PIL.Image")
    image = image.convert("RGB")
    label, confidence, _ = _predict_from_pil(image)
    return label, confidence

def predict_path(path: str) -> Dict[str, Union[str, float, int, list]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    img = Image.open(path).convert("RGB")
    return analyze(img, filename=os.path.basename(path))

def reset_counter_and_log(confirm: bool = False) -> None:
    if not confirm:
        raise RuntimeError("Passe confirm=True para executar.")
    _write_count(0)
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    _load_model()
    print("Modo:", current_mode())
    if current_mode() == "zeroshot":
        _build_text_features()
        print("Classes zero-shot:", ", ".join(CLASSES))
    print("Pronto.")




