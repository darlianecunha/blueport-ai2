# -*- coding: utf-8 -*-
"""
check_dataset.py — Verificador do dataset do BluePort
- Lista imagens válidas por classe
- Detecta arquivos inválidos/corrompidos/zero-byte
- Detecta .DS_Store e extensões não suportadas
- Gera relatório CSV (dataset_report.csv)
- Opção --clean: remove .DS_Store e move arquivos inválidos para quarantine/

Uso:
    python check_dataset.py --root ./dataset
    python check_dataset.py --root ./dataset --clean
"""

import os
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from PIL import Image, UnidentifiedImageError

SUPPORTED_EXT = {".jpg", ".jpeg", ".png"}
REPORT_CSV = "dataset_report.csv"

def is_hidden(p: Path) -> bool:
    return p.name.startswith(".") or p.name == "__MACOSX"

def try_open_image(path: Path) -> Tuple[bool, Optional[str], Optional[Tuple[int,int]]]:
    """Tenta abrir a imagem com PIL; retorna (ok, erro, (w,h))"""
    try:
        if path.stat().st_size == 0:
            return False, "zero-byte file", None
        with Image.open(path) as im:
            im.verify()  # checagem rápida de integridade
        # reabre para obter tamanho (depois do verify precisa reabrir)
        with Image.open(path) as im2:
            im2.load()
            return True, None, im2.size
    except UnidentifiedImageError:
        return False, "not an image / unidentified format", None
    except Exception as e:
        return False, f"PIL error: {type(e).__name__}: {e}", None

def scan_dataset(root: Path) -> Tuple[List[dict], Dict[str,int]]:
    rows: List[dict] = []
    counts: Dict[str,int] = {}
    for class_dir in sorted([d for d in root.iterdir() if d.is_dir() and not is_hidden(d)]):
        cls = class_dir.name
        counts.setdefault(cls, 0)
        for fp in class_dir.rglob("*"):
            if fp.is_dir():
                continue
            status = "ok"
            issue = ""
            size = ""
            width = height = ""
            ext = fp.suffix.lower()

            if is_hidden(fp):
                status = "hidden"
                issue = "hidden file (.DS_Store or dotfile)"
            elif ext not in SUPPORTED_EXT:
                status = "unsupported_ext"
                issue = f"unsupported extension {ext} (supported: {', '.join(sorted(SUPPORTED_EXT))})"
            else:
                ok, err, wh = try_open_image(fp)
                if ok:
                    counts[cls] += 1
                    width, height = wh
                    # alerta opcional: imagens muito pequenas
                    if wh[0] < 100 or wh[1] < 100:
                        status = "warn_small"
                        issue = "very small image (<100px)"
                else:
                    status = "broken"
                    issue = err or "unknown error"

            try:
                size = fp.stat().st_size
            except Exception:
                size = ""

            rows.append({
                "class": cls,
                "filepath": str(fp),
                "filename": fp.name,
                "ext": ext,
                "status": status,
                "issue": issue,
                "size_bytes": size,
                "width": width,
                "height": height,
            })
    return rows, counts

def write_report(rows: List[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["class","filepath","filename","ext","status","issue","size_bytes","width","height"]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

def clean_issues(rows: List[dict], root: Path, quarantine_dir: Path) -> Tuple[int,int]:
    """Remove .DS_Store/hidden e move inválidos para quarantine/"""
    removed = 0
    moved = 0
    quarantine_dir.mkdir(exist_ok=True, parents=True)

    for r in rows:
        fp = Path(r["filepath"])
        if r["status"] == "hidden" and fp.exists():
            try:
                fp.unlink()
                removed += 1
            except Exception:
                pass
        elif r["status"] in {"unsupported_ext","broken"} and fp.exists():
            # recria estrutura em quarantine mantendo subpastas/classes
            rel = fp.relative_to(root)
            dest = quarantine_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                fp.rename(dest)
                moved += 1
            except Exception:
                pass
    return removed, moved

def main():
    ap = argparse.ArgumentParser(description="Check & clean BluePort dataset")
    ap.add_argument("--root", required=True, help="raiz do dataset (ex.: ./dataset)")
    ap.add_argument("--clean", action="store_true", help="apaga .DS_Store e move arquivos inválidos para quarantine/")
    ap.add_argument("--report", default=REPORT_CSV, help=f"nome do CSV de saída (padrão: {REPORT_CSV})")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Pasta não encontrada: {root}")

    print(f"📂 Verificando dataset em: {root}")
    rows, counts = scan_dataset(root)
    write_report(rows, Path(args.report).resolve())

    total_ok = sum(1 for r in rows if r["status"] in {"ok","warn_small"})
    total_broken = sum(1 for r in rows if r["status"] == "broken")
    total_hidden = sum(1 for r in rows if r["status"] == "hidden")
    total_unsupported = sum(1 for r in rows if r["status"] == "unsupported_ext")

    print("—"*56)
    print("Resumo por classe:")
    for cls in sorted(counts):
        print(f"  • {cls}: {counts[cls]} imagens válidas")
    print("—"*56)
    print(f"✅ Válidas: {total_ok}")
    print(f"⚠️ Hidden (.DS_Store/dotfiles): {total_hidden}")
    print(f"⚠️ Extensão não suportada: {total_unsupported}")
    print(f"❌ Corrompidas / não reconhecidas: {total_broken}")
    print(f"📝 Relatório CSV salvo em: {Path(args.report).resolve()}")

    if args.clean:
        removed, moved = clean_issues(rows, root, root.parent / "quarantine")
        print("—"*56)
        print(f"🧹 Limpeza feita: removidos {removed} hidden, movidos {moved} inválidos → {root.parent / 'quarantine'}")

if __name__ == "__main__":
    main()
