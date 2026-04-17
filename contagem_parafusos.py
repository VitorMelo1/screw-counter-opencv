"""
Contagem de parafusos / peças em imagem com fundo claro.

Pipeline (conceitos da aula: binarização + morfologia):
  1) Pré-processamento em tons de cinza e suavização leve.
  2) Binarização (Otsu) — objetos escuros sobre fundo branco.
  3) Fechamento (dilatação → erosão): preenche buracos internos (reflexos nas peças).
  4) Abertura (erosão → dilatação): remove ruídos finos e reduz “pontes” entre sombras.
  5) Rotulagem de componentes conexos com filtro por área mínima.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def preprocess(gray: np.ndarray, blur_ksize: int = 5) -> np.ndarray:
    if blur_ksize >= 3 and blur_ksize % 2 == 1:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    return gray


def binarize_otsu(gray: np.ndarray) -> tuple[np.ndarray, int]:
    """Retorna máscara com primeiro plano = 255 (objetos escuros)."""
    t, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return binary, int(t)


def morph_close_then_open(
    mask: np.ndarray,
    close_ksize: int,
    open_ksize: int,
) -> np.ndarray:
    """Fechamento depois abertura (ordem típica para sólidos com ruído fino)."""
    out = mask.copy()
    if close_ksize >= 3:
        kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kc)
    if open_ksize >= 3:
        ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, ko)
    return out


def count_components(
    mask_fg: np.ndarray,
    min_area: int,
    max_area: int | None = None,
) -> tuple[int, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """
    Conta componentes 8-conectados com área em [min_area, max_area].
    Retorna: (quantidade, labels coloridos BGR, stats, centroides aproximados).
    """
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_fg, connectivity=8
    )
    h, w = mask_fg.shape[:2]
    total_pixels = h * w
    if max_area is None:
        max_area = int(0.25 * total_pixels)

    valid = 0
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    centers: list[tuple[int, int]] = []
    rng = np.random.default_rng(42)
    colors = rng.integers(50, 255, size=(num, 3), dtype=np.int32)

    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        valid += 1
        color = tuple(int(c) for c in colors[i])
        vis[labels == i] = color
        cx, cy = int(centroids[i, 0]), int(centroids[i, 1])
        centers.append((cx, cy))

    return valid, vis, stats, centers


def process_image(
    path: Path,
    close_ksize: int = 9,
    open_ksize: int = 5,
    min_area_ratio: float = 0.0008,
    max_area_ratio: float | None = None,
    blur_ksize: int = 5,
    save_debug: Path | None = None,
) -> dict:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = preprocess(gray, blur_ksize)
    binary, thr = binarize_otsu(gray)
    cleaned = morph_close_then_open(binary, close_ksize, open_ksize)

    h, w = gray.shape
    min_area = max(80, int(min_area_ratio * h * w))
    max_area = None
    if max_area_ratio is not None:
        max_area = int(max_area_ratio * h * w)
    count, vis, stats, centers = count_components(
        cleaned, min_area=min_area, max_area=max_area
    )

    result = {
        "path": path,
        "otsu_threshold": thr,
        "count": count,
        "min_area_used": min_area,
        "max_area_used": max_area,
    }

    if save_debug is not None:
        save_debug.mkdir(parents=True, exist_ok=True)
        stem = path.stem
        cv2.imwrite(str(save_debug / f"{stem}_1_gray.jpg"), gray)
        cv2.imwrite(str(save_debug / f"{stem}_2_binary_otsu.jpg"), binary)
        cv2.imwrite(str(save_debug / f"{stem}_3_morfologia.jpg"), cleaned)
        cv2.imwrite(str(save_debug / f"{stem}_4_rotulos.jpg"), vis)
        overlay = cv2.addWeighted(bgr, 0.55, vis, 0.45, 0)
        for cx, cy in centers:
            cv2.drawMarker(
                overlay, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 14, 2
            )
        cv2.putText(
            overlay,
            f"Contagem: {count}",
            (24, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 180, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(save_debug / f"{stem}_5_overlay.jpg"), overlay)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Contagem de parafusos com morfologia.")
    parser.add_argument(
        "imagens",
        nargs="*",
        type=Path,
        help="Caminhos das imagens (default: PARAFUSOS *.jpg na pasta do script).",
    )
    parser.add_argument("--close", type=int, default=9, help="Tamanho do kernel de fechamento (ímpar >=3).")
    parser.add_argument("--open", type=int, default=5, help="Tamanho do kernel de abertura (ímpar >=3).")
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.0008,
        help="Área mínima = ratio * largura * altura (filtra ruído).",
    )
    parser.add_argument(
        "--max-area-ratio",
        type=float,
        default=None,
        help="Opcional: área máxima = ratio * W * H (exclui objetos muito grandes).",
    )
    parser.add_argument("--blur", type=int, default=5, help="Gaussian blur (ímpar >=3 ou 0).")
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="Se definido, salva etapas intermediárias nesta pasta.",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    if args.imagens:
        paths = args.imagens
    else:
        paths = sorted(base.glob("PARAFUSOS*.jpg"))
        if not paths:
            paths = sorted(base.glob("*.jpg"))

    if not paths:
        raise SystemExit("Nenhuma imagem encontrada. Passe caminhos ou coloque JPGs na pasta.")

    for p in paths:
        r = process_image(
            p,
            close_ksize=args.close,
            open_ksize=args.open,
            min_area_ratio=args.min_area_ratio,
            max_area_ratio=args.max_area_ratio,
            blur_ksize=args.blur,
            save_debug=args.debug_dir,
        )
        extra = f", min_area={r['min_area_used']}"
        if r["max_area_used"] is not None:
            extra += f", max_area={r['max_area_used']}"
        print(f"{r['path'].name}: contagem = {r['count']} (Otsu T={r['otsu_threshold']}{extra})")


if __name__ == "__main__":
    main()
