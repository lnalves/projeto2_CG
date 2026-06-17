"""
Orquestra o pipeline de medição de plântulas de alface.
Pipeline: carregar → calibrar → pré-processar → segmentar → esqueletizar
→ detectar pontos → medir → anotar/exportar.

Uso:
    python src/main.py --image data/exemplo.jpg --out output
"""

from __future__ import annotations

import argparse
import sys

from config import default_config


def build_parser() -> argparse.ArgumentParser:
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


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = default_config()
    config.debug = args.debug
    if args.known_mm is not None:
        config.calibration.known_distance_mm = args.known_mm

    if not args.image:
        parser.error("informe --image com o caminho da foto.")


if __name__ == "__main__":
    sys.exit(main())
