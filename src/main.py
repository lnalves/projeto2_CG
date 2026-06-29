"""Orquestra a medição semiautomática de plântulas de alface (live-wire).

Pipeline: carregar → calibrar (2 cliques com distância conhecida) → pré-processar
→ medir por cliques (topo→ponta, caminho de custo mínimo) → anotar/exportar.

Uso:
    python src/main.py --image data/IMG_3196.png --known-mm 90 --out output
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

from calibration import calibrar_detalhado
from config import config_padrao
from livewire import ferramenta_livewire
from preprocess import preprocessar
from render import anotar_imagem, exportar_tabela


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Mede plântulas de alface (hipocótilo + raiz) a partir de uma foto, em mm.",
    )
    parser.add_argument(
        "--image", "-i",
        help="Caminho da imagem de entrada (foto das plântulas com a régua).",
    )
    parser.add_argument(
        "--out", "-o",
        default="output",
        help="Pasta de saída para imagem anotada e tabela (padrão: output).",
    )
    parser.add_argument(
        "--known-mm",
        type=float,
        default=90.0,
        help="Distância real (mm) entre os 2 pontos da calibração manual (padrão: 90).",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=1600,
        help="Redimensiona a imagem p/ que o maior lado tenha no máx. N px (0 desliga).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Salva imagens intermediárias em <out>/debug/.",
    )
    return parser


def carregar_imagem(caminho):
    """Carrega a imagem como BGR. Suporta HEIC/HEIF se `pillow-heif` existir."""
    ext = os.path.splitext(caminho)[1].lower()
    if ext in (".heic", ".heif"):
        try:
            from pillow_heif import register_heif_opener
            from PIL import Image
        except ImportError:
            raise RuntimeError(
                f"'{caminho}' é HEIC e 'pillow-heif' não está instalado. "
                "Instale com `pip install pillow-heif` ou converta a foto para PNG/JPG."
            )
        register_heif_opener()
        pil = Image.open(caminho).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    img = cv2.imread(caminho, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"não foi possível ler a imagem '{caminho}'.")
    return img


def redimensionar(imagem, max_dim):
    """Reduz a imagem para que o maior lado tenha no máx. `max_dim`.

    Retorna (imagem, escala). A escala (<=1) multiplica coordenadas da imagem
    original para chegar nas da imagem reduzida.
    """
    if not max_dim or max_dim <= 0:
        return imagem, 1.0
    h, w = imagem.shape[:2]
    maior = max(h, w)
    if maior <= max_dim:
        return imagem, 1.0
    s = max_dim / float(maior)
    nova = cv2.resize(imagem, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    return nova, s


def processar(imagem, config, regua_roi, dir_debug=None):
    """Roda o pipeline e devolve (imagem_anotada, resultados).

    `resultados` é uma lista de `ResultadoPlantula` coletada por cliques
    (live-wire, ADR 0001).
    """
    # Calibração (régua → mm/px).
    cal = calibrar_detalhado(imagem, config, regua_roi=regua_roi)
    print(
        f"Calibração [{cal.metodo}] confiança={cal.confianca:.2f} | "
        f"escala={cal.mm_por_px:.5f} mm/px ({1.0 / cal.mm_por_px:.2f} px/mm)"
    )

    # Pré-processamento → canais de realce.
    canais = preprocessar(imagem, config, dir_debug=dir_debug)

    # Medição por cliques (topo→ponta, caminho de custo mínimo).
    resultados = ferramenta_livewire(imagem, canais, cal.mm_por_px, config)

    # Imagem anotada.
    anotada = anotar_imagem(imagem, resultados, config)
    return anotada, resultados


def main(argv=None) -> int:
    parser = construir_parser()
    args = parser.parse_args(argv)

    config = config_padrao()
    config.debug = args.debug
    config.calibracao.distancia_conhecida_mm = args.known_mm

    if not args.image:
        parser.error("informe --image com o caminho da foto.")

    try:
        imagem = carregar_imagem(args.image)
    except RuntimeError as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 1

    # Redimensiona imagens grandes.
    imagem, escala = redimensionar(imagem, args.max_dim)
    if escala != 1.0:
        print(f"Imagem redimensionada (escala={escala:.3f}) → {imagem.shape[1]}x{imagem.shape[0]}")

    dir_debug = os.path.join(args.out, "debug") if args.debug else None

    try:
        anotada, resultados = processar(imagem, config, regua_roi=None, dir_debug=dir_debug)
    except RuntimeError as exc:
        print(f"erro no pipeline: {exc}", file=sys.stderr)
        return 1

    # --- Etapa 7: salvar saídas ---
    os.makedirs(args.out, exist_ok=True)
    nome_base = os.path.splitext(os.path.basename(args.image))[0]
    caminho_img = os.path.join(args.out, f"{nome_base}_anotada.png")
    caminho_csv = os.path.join(args.out, f"{nome_base}_medidas.csv")

    cv2.imwrite(caminho_img, anotada)
    df = exportar_tabela(resultados, caminho_csv)

    print("\n" + df.to_string(index=False) if not df.empty else "\n(nenhuma plântula medida)")
    print(f"\nImagem anotada: {caminho_img}")
    print(f"Tabela CSV:     {caminho_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
