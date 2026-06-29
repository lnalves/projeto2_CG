"""Saída: imagem anotada + tabela.

Desenha, para cada plântula:
  - o caminho do **segmento 1** (hipocótilo) e do **segmento 2** (raiz) em cores
    distintas (`cv2.polylines`);
  - os **três pontos** (topo, colo, ponta) com `cv2.circle`;
  - um rótulo `#id: total mm` com `cv2.putText`.

E exporta a tabela de medidas (`pandas` → CSV) com as colunas
`id, segmento1_mm, segmento2_mm, total_mm`.

`ResultadoPlantula` é a unidade que o `main.py` monta por plântula e entrega
aqui — agrega o id, os três pontos (y, x) e a medição.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd

# Cores fixas dos pontos (BGR), para leitura clara independentemente das cores
# dos segmentos configuradas em `config.renderizacao`.
_COR_TOPO = (0, 220, 0)      # verde  — topo da estrutura branca
_COR_COLO = (0, 0, 255)      # vermelho — colo (estrangulamento)
_COR_PONTA = (255, 0, 0)     # azul   — ponta da raiz


@dataclass
class ResultadoPlantula:
    """Tudo que descreve uma plântula medida (para anotar e tabelar)."""

    id: int
    topo: tuple          # (y, x)
    colo: tuple          # (y, x)
    ponta: tuple         # (y, x)
    medicao: object      # measure.ResultadoMedicao


def _para_xy(caminho_yx):
    """Converte [(y, x), ...] no array (N,1,2) em (x, y) que o OpenCV espera."""
    if not caminho_yx:
        return None
    pts = np.array([[x, y] for (y, x) in caminho_yx], dtype=np.int32)
    return pts.reshape(-1, 1, 2)


def anotar_imagem(imagem, resultados, config) -> np.ndarray:
    """Desenha caminhos, pontos e rótulos de todas as plântulas sobre a imagem.
    Retorna uma cópia anotada
    (a imagem original não é modificada).
    """
    ren = config.renderizacao
    saida = imagem.copy()
    if saida.ndim == 2:
        saida = cv2.cvtColor(saida, cv2.COLOR_GRAY2BGR)

    for r in resultados:
        med = r.medicao

        # Caminhos dos dois segmentos.
        p1 = _para_xy(med.caminho1)
        if p1 is not None:
            cv2.polylines(saida, [p1], False, ren.cor_seg1_bgr, ren.espessura)
        p2 = _para_xy(med.caminho2)
        if p2 is not None:
            cv2.polylines(saida, [p2], False, ren.cor_seg2_bgr, ren.espessura)

        # Três pontos (y, x) → (x, y).
        ty, tx = r.topo
        cy, cx = r.colo
        py, px = r.ponta
        cv2.circle(saida, (tx, ty), ren.raio_ponto, _COR_TOPO, -1)
        cv2.circle(saida, (cx, cy), ren.raio_ponto, _COR_COLO, -1)
        cv2.circle(saida, (px, py), ren.raio_ponto, _COR_PONTA, -1)

        # Rótulo com id e total, próximo ao topo.
        texto = f"#{r.id}: {med.total_mm:.1f}mm"
        cv2.putText(
            saida, texto, (tx + 8, ty - 8),
            cv2.FONT_HERSHEY_SIMPLEX, ren.escala_fonte, _COR_TOPO, ren.espessura,
            cv2.LINE_AA,
        )

    return saida


def montar_tabela(resultados) -> pd.DataFrame:
    """DataFrame com id e medidas (mm) de cada plântula."""
    linhas = [
        {
            "id": r.id,
            "segmento1_mm": round(r.medicao.segmento1_mm, 2),
            "segmento2_mm": round(r.medicao.segmento2_mm, 2),
            "total_mm": round(r.medicao.total_mm, 2),
        }
        for r in resultados
    ]
    return pd.DataFrame(linhas, columns=["id", "segmento1_mm", "segmento2_mm", "total_mm"])


def exportar_tabela(resultados, caminho_saida) -> pd.DataFrame:
    """Exporta a tabela de medidas em CSV e a devolve como DataFrame."""
    df = montar_tabela(resultados)
    df.to_csv(caminho_saida, index=False)
    return df
