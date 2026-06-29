"""Calibração da escala (mm por pixel) a partir da régua na cena.

Apenas modo manual: o usuário clica em dois pontos de distância real
conhecida (distancia_conhecida_mm) e a escala sai da razão.

A função pública `calibrar` devolve a escala em **mm/px** (multiplicar um
comprimento em pixels por essa escala dá o comprimento em milímetros).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ResultadoCalibracao:
    """Resultado detalhado da calibração."""

    mm_por_px: float
    metodo: str          # "manual"
    confianca: float     # 1.0
    px_por_unidade: float


def calibrar(imagem, config, regua_roi=None) -> float:
    """Retorna a escala mm/px via calibração manual (2 cliques)."""
    resultado = calibrar_detalhado(imagem, config, regua_roi=regua_roi)
    return resultado.mm_por_px


def calibrar_detalhado(imagem, config, regua_roi=None) -> ResultadoCalibracao:
    """Executa calibração manual por 2 cliques. `regua_roi` é ignorado."""
    manual = _calibrar_manual(imagem, config.calibracao)
    if manual is None:
        raise RuntimeError(
            "Calibração falhou: nenhum ponto foi marcado manualmente."
        )
    return manual


def _calibrar_manual(imagem, cal) -> ResultadoCalibracao | None:
    """Coleta dois cliques e converte a distância conhecida em mm/px."""
    pontos = _dois_cliques(imagem, cal)
    if pontos is None or len(pontos) < 2:
        return None

    (x0, y0), (x1, y1) = pontos[0], pontos[1]
    dist_px = float(np.hypot(x1 - x0, y1 - y0))
    if dist_px <= 0:
        return None

    mm_por_px = cal.distancia_conhecida_mm / dist_px
    return ResultadoCalibracao(
        mm_por_px=mm_por_px,
        metodo="manual",
        confianca=1.0,
        px_por_unidade=dist_px,
    )


def _dois_cliques(imagem, cal) -> list[tuple[int, int]] | None:
    exibicao, escala = _ajustar_exibicao(imagem, cal.largura_max_exibicao)
    janela = "Calibracao: clique 2 pontos (dist. conhecida) | ENTER ok, ESC cancela"
    cliques: list[tuple[int, int]] = []

    def ao_clicar(evento, mx, my, flags, param):
        if evento == cv2.EVENT_LBUTTONDOWN and len(cliques) < 2:
            # Volta para coordenadas da imagem original.
            cliques.append((int(round(mx / escala)), int(round(my / escala))))

    cv2.namedWindow(janela, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(janela, ao_clicar)
    try:
        while True:
            tela = exibicao.copy()
            for i, (px, py) in enumerate(cliques):
                dp = (int(round(px * escala)), int(round(py * escala)))
                cv2.circle(tela, dp, 5, (0, 0, 255), -1)
                cv2.putText(
                    tela, str(i + 1), (dp[0] + 8, dp[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                )
            if len(cliques) == 2:
                p0 = (int(round(cliques[0][0] * escala)), int(round(cliques[0][1] * escala)))
                p1 = (int(round(cliques[1][0] * escala)), int(round(cliques[1][1] * escala)))
                cv2.line(tela, p0, p1, (0, 0, 255), 2)

            cv2.imshow(janela, tela)
            tecla = cv2.waitKey(20) & 0xFF
            if tecla == 27:  # ESC
                cliques = []
                break
            if tecla in (13, 10) and len(cliques) == 2:  # ENTER
                break
    finally:
        cv2.destroyWindow(janela)

    return cliques if len(cliques) == 2 else None


def _ajustar_exibicao(imagem, largura_max):
    """Reduz a imagem para caber na largura máxima de exibição."""
    h, w = imagem.shape[:2]
    if w <= largura_max:
        return imagem, 1.0
    escala = largura_max / float(w)
    exibicao = cv2.resize(
        imagem, (largura_max, int(round(h * escala))), interpolation=cv2.INTER_AREA
    )
    return exibicao, escala
