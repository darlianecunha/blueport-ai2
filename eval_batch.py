# -*- coding: utf-8 -*-
"""
eval_batch.py — Avaliação em lote para o BluePort AI
Gera um CSV com: caminho, arquivo, classe_prevista, confiança, top3,
e (opcional) classe_verdadeira + acerto quando a estrutura for dataset/<classe>/arquivo.jpg.

Uso:
    python eval_batch.py --input ./dataset --output results.csv
    python eval_batch.py --input ./minhas_fotos --output results.csv
"""

import os
import csv
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

from PIL import Image
from tqdm import tqdm

# Importa o núcleo. Vamos usar a função interna _predict_from_pil (sem efeitos colaterais)
import waste_vision


SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(root: Path):
    """Itera recursivamente por imagens em root."""
    for p in root.rglob("*"):
        if p.suffix.lower() in SUPPORTED_EXT and p.is_file():
            yield p


def infer_true_label(path: Path, dataset_root: Path) -> Optional[str]:
    """
    Se a estrutura for dataset/<classe>/<arquivo>, retorna <classe> como rótulo verdadeiro.
    Caso contrário, retorna None.
    """
    try:
        # dataset_root / class_dir / file
        rel = path.relative_to(dataset_root)
        parts = rel.parts
        if len(parts) >= 2:
            return parts[0]
    except Exception:
        pass
    return None


def predict_image(path: Path) -> Tuple[str, float, List[Tuple[str, float]]]:
    """
    Usa a função interna do waste_vision que devolve (label, confidence, top3)
    sem mexer em contador nem log.
    """
    img = Image.open(path).convert("RGB")
    label, conf, top3 = waste_vision._predict_from_pil(img)  # noqa: SLF001 (acesso a função "privada")
    return label, conf, top3


def main():
    parser = argparse.ArgumentParser(description="Avaliação em lote do BluePort AI")
    parser.add_argument("--input", required=True, help="Pasta de entrada com imagens (ex.: ./dataset ou ./fotos)")
    parser.add_argument("--output", default="results.csv", help="Arquivo CSV de saída (padrão: results.csv)")
    args = parser.parse_args()

    in_dir = Path(args.input).resolve()
    out_csv = Path(args.output).resolve()

    if not in_dir.exists() or not in_dir.is_dir():
        raise SystemExit(f"Pasta de entrada não existe: {in_dir}")

    # Garante que o modelo esteja carregado (e o linear probe, se existir)
    waste_vision._load_model()  # carrega CLIP e, se houver, blueport_linear.pt

    rows = []
    total = 0
    correct = 0

    for img_path in tqdm(list(iter_images(in_dir)), desc="Avaliando imagens"):
        try:
            pred_label, conf, top3 = predict_image(img_path)

            # tenta descobrir classe verdadeira (se estrutura for dataset/<classe>/arquivo)
            true_label = infer_true_label(img_path, in_dir)
            is_correct = None
            if true_label is not None:
                is_correct = (pred_label == true_label)
                if is_correct:
                    correct += 1
                total += 1

            # normaliza top3 em colunas
            t1_l, t1_c = (top3[0][0], float(top3[0][1])) if len(top3) > 0 else ("", 0.0)
            t2_l, t2_c = (top3[1][0], float(top3[1][1])) if len(top3) > 1 else ("", 0.0)
            t3_l, t3_c = (top3[2][0], float(top3[2][1])) if len(top3) > 2 else ("", 0.0)

            rows.append({
                "filepath": str(img_path),
                "filename": img_path.name,
                "pred_label": pred_label,
                "confidence": round(float(conf), 6),
                "top1_label": t1_l, "top1_conf": round(t1_c, 6),
                "top2_label": t2_l, "top2_conf": round(t2_c, 6),
                "top3_label": t3_l, "top3_conf": round(t3_c, 6),
                "true_label": true_label if true_label is not None else "",
                "correct": "" if is_correct is None else int(is_correct),
            })
        except Exception as e:
            rows.append({
                "filepath": str(img_path),
                "filename": img_path.name,
                "pred_label": "ERROR",
                "confidence": 0.0,
                "top1_label": "", "top1_conf": 0.0,
                "top2_label": "", "top2_conf": 0.0,
                "top3_label": "", "top3_conf": 0.0,
                "true_label": "",
                "correct": "",
                "error": str(e),
            })

    # escreve CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filepath", "filename",
                "pred_label", "confidence",
                "top1_label", "top1_conf",
                "top2_label", "top2_conf",
                "top3_label", "top3_conf",
                "true_label", "correct", "error"
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # imprime acurácia se houve rótulo verdadeiro
    if total > 0:
        acc = 100.0 * correct / total
        print(f"✅ Avaliação concluída. Acurácia: {acc:.1f}% ({correct}/{total})")
    else:
        print("✅ Avaliação concluída. (Sem rótulos verdadeiros para calcular acurácia)")

    print(f"CSV salvo em: {out_csv}")


if __name__ == "__main__":
    main()
