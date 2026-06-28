"""Orquestra o pipeline de medição de plântulas de alface.

Pipeline: carregar → calibrar → pré-processar → segmentar → esqueletizar
→ detectar pontos → medir → anotar/exportar.

Uso:
    python src/main.py --image data/exemplo.jpg --out output
    python src/main.py -i data/exemplo.jpg --ruler-roi 10,10,400,40 --known-mm 90
    python src/main.py -i data/exemplo.jpg --tune-seg   # ajusta segmentação ao vivo
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

from calibration import calibrar_detalhado
from config import config_padrao
from measure import medir_segmentos
from preprocess import preprocessar
from render import ResultadoPlantula, anotar_imagem, exportar_tabela
from segmentation import ajustar_segmentacao, segmentar
from skeleton import detectar_pontos, esqueletizar_plantula


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
        "--tune-seg",
        action="store_true",
        help="Abre trackbars para ajustar a segmentação ao vivo antes de medir.",
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


def processar(imagem, config, regua_roi, ajustar_seg):
    """Roda o pipeline completo e devolve (imagem_anotada, resultados).

    `resultados` é uma lista de `ResultadoPlantula`.
    """
    # --- Etapa 1: calibração (régua → mm/px) ---
    cal = calibrar_detalhado(imagem, config, regua_roi=regua_roi)
    print(
        f"Calibração [{cal.metodo}] confiança={cal.confianca:.2f} | "
        f"escala={cal.mm_por_px:.5f} mm/px ({1.0 / cal.mm_por_px:.2f} px/mm)"
    )

    # --- Etapa 2: pré-processamento ---
    dir_debug = None
    canais = preprocessar(imagem, config, dir_debug=dir_debug)

    # --- Etapa 3: segmentação (com ajuste opcional por trackbars) ---
    seg = ajustar_segmentacao(canais, config) if ajustar_seg else segmentar(canais, config)
    print(f"Plântulas detectadas: {seg.num_plantulas}")

    # --- Etapas 4–6: por plântula → esqueleto, pontos e medida ---
    resultados = []
    for pid in range(1, seg.num_plantulas + 1):
        mascara_p = np.where(seg.rotulos == pid, 255, 0).astype(np.uint8)
        if int((mascara_p > 0).sum()) < config.segmentacao.area_minima_px:
            continue
        semente_p = cv2.bitwise_and(seg.mascara_semente, mascara_p)

        res_esq = esqueletizar_plantula(mascara_p, config)
        if not (res_esq.esqueleto > 0).any():
            print(f"  plântula {pid}: esqueleto vazio, ignorada.", file=sys.stderr)
            continue
        try:
            topo, colo, ponta = detectar_pontos(
                res_esq.esqueleto, res_esq.mapa_distancia, semente_p, config
            )
            med = medir_segmentos(
                res_esq.esqueleto, topo, colo, ponta, cal.mm_por_px, config
            )
        except (ValueError, RuntimeError) as exc:
            print(f"  plântula {pid}: ignorada ({exc}).", file=sys.stderr)
            continue

        resultados.append(
            ResultadoPlantula(id=len(resultados) + 1, topo=topo, colo=colo,
                              ponta=ponta, medicao=med)
        )

    # --- Etapa 7: imagem anotada ---
    anotada = anotar_imagem(imagem, resultados, config)
    return anotada, resultados


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

    try:
        anotada, resultados = processar(imagem, config, regua_roi, args.tune_seg)
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
