"""Orquestra o pipeline de medição de plântulas de alface.

Pipeline: carregar → calibrar → pré-processar → segmentar → esqueletizar
→ detectar pontos → medir → anotar/exportar.

Uso:
    python src/main.py --image data/exemplo.jpg --out output
"""

from __future__ import annotations

import argparse
import sys

import cv2

from calibration import calibrar_detalhado
from config import config_padrao


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
        default=None,
        help="Distância real (mm) entre os 2 pontos da calibração manual.",
    )
    parser.add_argument(
        "--ruler-roi",
        default=None,
        help="ROI da régua para detecção automática: x,y,w,h.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Salva imagens intermediárias em <out>/debug/.",
    )
    return parser


def analisar_roi(texto):
    """Converte 'x,y,w,h' em uma tupla de inteiros. Erro claro se inválido."""
    if texto is None:
        return None
    partes = texto.split(",")
    if len(partes) != 4:
        raise ValueError("--ruler-roi deve ter 4 valores: x,y,w,h")
    try:
        x, y, w, h = (int(p.strip()) for p in partes)
    except ValueError:
        raise ValueError("--ruler-roi deve conter inteiros: x,y,w,h")
    if w <= 0 or h <= 0:
        raise ValueError("--ruler-roi precisa de largura e altura positivas")
    return (x, y, w, h)


def main(argv=None) -> int:
    parser = construir_parser()
    args = parser.parse_args(argv)

    config = config_padrao()
    config.debug = args.debug
    if args.known_mm is not None:
        config.calibracao.distancia_conhecida_mm = args.known_mm

    if not args.image:
        parser.error("informe --image com o caminho da foto.")

    try:
        regua_roi = analisar_roi(args.ruler_roi)
    except ValueError as exc:
        parser.error(str(exc))

    imagem = cv2.imread(args.image)
    if imagem is None:
        print(f"erro: não foi possível ler a imagem '{args.image}'.", file=sys.stderr)
        return 1

    # --- Etapa 1: calibração (régua → mm/px) ---
    try:
        cal = calibrar_detalhado(imagem, config, regua_roi=regua_roi)
    except RuntimeError as exc:
        print(f"erro na calibração: {exc}", file=sys.stderr)
        return 1

    print(
        f"Calibração [{cal.metodo}] confiança={cal.confianca:.2f} | "
        f"escala={cal.mm_por_px:.5f} mm/px "
        f"({1.0 / cal.mm_por_px:.2f} px/mm)"
    )

    # TODO: pré-processar → segmentar → esqueletizar → medir → anotar/exportar.
    return 0


if __name__ == "__main__":
    sys.exit(main())
