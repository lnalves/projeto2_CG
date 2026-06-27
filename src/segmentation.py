"""Etapa 3 — Segmentação + separação das plântulas.

Ponto mais sensível do pipeline: a plântula é esbranquiçada sobre papel branco.
A estratégia combina duas pistas, conforme o plano:

  - **Semente escura** (`máscara_semente`): threshold baixo sobre a luminância
    (canal V) — a semente/cotilédones é a parte mais escura/amarelada no topo.
  - **Filamento** (hipocótilo + raiz): threshold adaptativo sobre o cinza+CLAHE,
    captando o desvio local de intensidade do filamento em relação ao fundo.

As duas máscaras são unidas, limpas por morfologia (OPEN remove ruído, CLOSE
fecha pequenas quebras no filamento) e rotuladas por componentes conexos.
Componentes abaixo de `area_minima_px` são descartados (sujeira/ruído).

Como ainda não há imagens reais, `ajustar_segmentacao` abre trackbars do OpenCV
para calibrar os parâmetros ao vivo; `segmentar` roda de forma não-interativa
usando os valores de `config.segmentacao`.

> Limitação conhecida: plântulas que se cruzam/encostam podem cair num único
> componente conexo. A separação manual desses casos fica para uma etapa futura.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ResultadoSegmentacao:
    """Saída da segmentação."""

    mascara: np.ndarray          # 0/255 uint8: plântulas filtradas por área
    mascara_semente: np.ndarray  # 0/255 uint8: sementes escuras
    rotulos: np.ndarray          # int32: 0=fundo, 1..N cada plântula
    num_plantulas: int           # quantidade de componentes válidos
    estatisticas: np.ndarray     # stats por rótulo (connectedComponentsWithStats)
    centroides: np.ndarray       # centróides por rótulo


def _impar(n: int, minimo: int = 1) -> int:
    """Garante valor ímpar >= `minimo` (exigido por adaptiveThreshold/kernels)."""
    n = max(minimo, int(n))
    return n if n % 2 == 1 else n + 1


def _pipeline_segmentacao(canais, seg) -> ResultadoSegmentacao:
    """Núcleo da segmentação a partir dos canais (Etapa 2) e dos parâmetros."""
    cinza = canais.cinza
    valor = canais.valor

    # 1) Semente escura: pixels mais escuros que o limiar na luminância.
    _, mascara_semente = cv2.threshold(
        valor, int(seg.limiar_semente), 255, cv2.THRESH_BINARY_INV
    )

    # 2) Filamento: threshold adaptativo captando o desvio local de intensidade.
    bloco = _impar(seg.bloco_adaptativo, minimo=3)
    filamento = cv2.adaptiveThreshold(
        cinza, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        bloco, int(seg.c_adaptativo),
    )

    # 3) União das pistas.
    bruta = cv2.bitwise_or(mascara_semente, filamento)

    # 4) Morfologia: OPEN remove ruído pontual, CLOSE fecha quebras no filamento.
    k = max(1, int(seg.kernel_morfologico))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    limpa = cv2.morphologyEx(
        bruta, cv2.MORPH_OPEN, kernel, iterations=max(0, int(seg.iter_abertura))
    )
    limpa = cv2.morphologyEx(
        limpa, cv2.MORPH_CLOSE, kernel, iterations=max(0, int(seg.iter_fechamento))
    )

    # 5) Componentes conexos + filtro por área mínima.
    num, rotulos, stats, centroides = cv2.connectedComponentsWithStats(
        limpa, connectivity=8
    )
    mascara_final = np.zeros_like(limpa)
    for lbl in range(1, num):  # 0 = fundo
        if stats[lbl, cv2.CC_STAT_AREA] >= int(seg.area_minima_px):
            mascara_final[rotulos == lbl] = 255

    # 6) Re-rotula só os componentes válidos → ids sequenciais 1..N.
    num_final, rotulos_final, stats_final, centroides_final = (
        cv2.connectedComponentsWithStats(mascara_final, connectivity=8)
    )

    # Limita a máscara de semente ao que sobreviveu (útil p/ achar o topo na Etapa 5).
    mascara_semente = cv2.bitwise_and(mascara_semente, mascara_final)

    return ResultadoSegmentacao(
        mascara=mascara_final,
        mascara_semente=mascara_semente,
        rotulos=rotulos_final,
        num_plantulas=max(0, num_final - 1),
        estatisticas=stats_final,
        centroides=centroides_final,
    )


def segmentar(canais, config) -> ResultadoSegmentacao:
    """Segmenta de forma não-interativa usando `config.segmentacao`."""
    return _pipeline_segmentacao(canais, config.segmentacao)


def colorir_rotulos(rotulos: np.ndarray) -> np.ndarray:
    """Gera uma imagem BGR colorindo cada plântula (para debug/visualização)."""
    saida = np.zeros((*rotulos.shape, 3), dtype=np.uint8)
    n = int(rotulos.max())
    if n == 0:
        return saida
    # Cores via matiz espaçada no HSV.
    matizes = np.linspace(0, 179, n, endpoint=False).astype(np.uint8)
    for lbl in range(1, n + 1):
        cor_hsv = np.uint8([[[matizes[lbl - 1], 200, 255]]])
        cor_bgr = cv2.cvtColor(cor_hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
        saida[rotulos == lbl] = cor_bgr
    return saida


# Ajuste interativo (trackbars)

_PARAMS_TRACKBAR = [
    # (nome, atributo em ConfigSegmentacao, valor_max)
    ("limiar_semente", "limiar_semente", 255),
    ("bloco_adaptativo", "bloco_adaptativo", 99),
    ("c_adaptativo", "c_adaptativo", 50),
    ("kernel_morf", "kernel_morfologico", 25),
    ("iter_abertura", "iter_abertura", 10),
    ("iter_fechamento", "iter_fechamento", 10),
    ("area_min/50", "area_minima_px", 100),  # escala 1 unidade = 50 px
]
_ESCALA_AREA = 50


def ajustar_segmentacao(canais, config) -> ResultadoSegmentacao:
    """Abre trackbars p/ calibrar a segmentação ao vivo sobre os canais dados.

    ENTER aceita os valores atuais (gravados em `config.segmentacao`) e devolve
    o resultado; ESC cancela e devolve a segmentação com os valores atuais.
    """
    seg = config.segmentacao
    janela = "Segmentacao: ajuste e ENTER p/ aceitar, ESC p/ cancelar"
    cv2.namedWindow(janela, cv2.WINDOW_NORMAL)

    for nome, attr, vmax in _PARAMS_TRACKBAR:
        if attr == "area_minima_px":
            inicial = int(getattr(seg, attr)) // _ESCALA_AREA
        else:
            inicial = int(getattr(seg, attr))
        cv2.createTrackbar(nome, janela, min(inicial, vmax), vmax, lambda _v: None)

    def _ler_trackbars_para(alvo):
        for nome, attr, _ in _PARAMS_TRACKBAR:
            v = cv2.getTrackbarPos(nome, janela)
            if attr == "area_minima_px":
                v *= _ESCALA_AREA
            setattr(alvo, attr, v)

    try:
        while True:
            _ler_trackbars_para(seg)
            res = _pipeline_segmentacao(canais, seg)
            cor = colorir_rotulos(res.rotulos)
            base = canais.bgr_suavizada
            tela = cv2.addWeighted(base, 0.6, cor, 0.4, 0)
            cv2.putText(
                tela, f"plantulas: {res.num_plantulas}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
            )
            cv2.imshow(janela, tela)
            tecla = cv2.waitKey(30) & 0xFF
            if tecla in (13, 10):  # ENTER → aceita
                break
            if tecla == 27:  # ESC → cancela
                break
    finally:
        cv2.destroyWindow(janela)

    return _pipeline_segmentacao(canais, seg)
