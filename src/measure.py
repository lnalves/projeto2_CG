"""Etapa 6 — Medição pelo caminho real + conversão de escala.

O esqueleto (Etapa 4) é modelado como um **grafo**: cada pixel é um nó, ligado
aos vizinhos do esqueleto com peso **1** (ortogonal) ou **√2** (diagonal). O
comprimento real de um trecho é o **menor caminho** (Dijkstra) entre dois
pontos — somando os pesos ao longo do caminho, o que acompanha curvas e partes
enroladas (e não a linha reta).

`medir_segmentos` mede os dois segmentos da plântula:
  - Segmento 1 (hipocótilo): topo → colo.
  - Segmento 2 (raiz):       colo → ponta.

Opcionalmente o caminho é suavizado por spline antes de medir, reduzindo o
serrilhado do esqueleto (que tende a superestimar o comprimento).

As funções de grafo (`construir_grafo`, `dijkstra_de`, `caminho_entre`) também
são usadas pela Etapa 5 (`skeleton.detectar_pontos`) para ordenar o caminho.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

try:
    from scipy.interpolate import splev, splprep
    _TEM_INTERP = True
except ImportError:
    _TEM_INTERP = False

# Deslocamentos dos 8 vizinhos e seus pesos (1 ortogonal, √2 diagonal).
_VIZINHOS_8 = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, sqrt(2)), (-1, 1, sqrt(2)), (1, -1, sqrt(2)), (1, 1, sqrt(2)),
]


@dataclass
class ResultadoMedicao:
    """Medidas dos dois segmentos da plântula (em mm) + caminhos percorridos."""

    segmento1_mm: float
    segmento2_mm: float
    total_mm: float
    caminho1: list = field(default_factory=list)  # [(y, x), ...] topo → colo
    caminho2: list = field(default_factory=list)  # [(y, x), ...] colo → ponta


# --- Grafo do esqueleto ----------------------------------------------------

def construir_grafo(esq_bool: np.ndarray):
    """Constrói o grafo esparso do esqueleto.

    Retorna (grafo_csr, pixels, idx_map):
      - grafo_csr: matriz esparsa NxN com os pesos das arestas;
      - pixels:    array (N, 2) com as coordenadas (y, x) de cada nó;
      - idx_map:   dict (y, x) -> índice do nó.
    """
    pixels = np.argwhere(esq_bool)
    idx_map = {(int(y), int(x)): i for i, (y, x) in enumerate(pixels)}
    h, w = esq_bool.shape

    linhas, colunas, pesos = [], [], []
    for i, (y, x) in enumerate(pixels):
        for dy, dx, peso in _VIZINHOS_8:
            ny, nx = int(y) + dy, int(x) + dx
            if 0 <= ny < h and 0 <= nx < w and esq_bool[ny, nx]:
                j = idx_map[(ny, nx)]
                linhas.append(i)
                colunas.append(j)
                pesos.append(peso)

    n = len(pixels)
    grafo = csr_matrix((pesos, (linhas, colunas)), shape=(n, n))
    return grafo, pixels, idx_map


def dijkstra_de(grafo, idx_origem):
    """Dijkstra a partir de um nó. Retorna (distancias, predecessores)."""
    dist, pred = dijkstra(
        grafo, directed=False, indices=idx_origem, return_predecessors=True
    )
    return dist, pred


def reconstruir_caminho(pred, idx_origem, idx_destino, pixels):
    """Reconstrói a lista de pixels (y, x) do caminho origem → destino."""
    if idx_destino < 0 or pred[idx_destino] == -9999 and idx_destino != idx_origem:
        return []
    caminho_idx = []
    atual = idx_destino
    # Limite de segurança contra inconsistências.
    for _ in range(len(pixels) + 1):
        caminho_idx.append(atual)
        if atual == idx_origem:
            break
        atual = pred[atual]
        if atual == -9999:
            return []  # sem caminho conectado
    caminho_idx.reverse()
    return [(int(pixels[i][0]), int(pixels[i][1])) for i in caminho_idx]


def caminho_entre(grafo, pixels, idx_map, origem_yx, destino_yx):
    """Caminho (lista de (y, x)) e comprimento em px entre dois pontos do grafo."""
    o = idx_map.get((int(origem_yx[0]), int(origem_yx[1])))
    d = idx_map.get((int(destino_yx[0]), int(destino_yx[1])))
    if o is None or d is None:
        return [], 0.0
    dist, pred = dijkstra_de(grafo, o)
    if not np.isfinite(dist[d]):
        return [], 0.0
    caminho = reconstruir_caminho(pred, o, d, pixels)
    return caminho, float(dist[d])


# --- Comprimento ao longo do caminho ---------------------------------------

def _comprimento_poligonal(caminho_yx) -> float:
    """Soma das distâncias euclidianas entre pixels consecutivos do caminho."""
    if len(caminho_yx) < 2:
        return 0.0
    pts = np.asarray(caminho_yx, dtype=np.float64)
    difs = np.diff(pts, axis=0)
    return float(np.hypot(difs[:, 0], difs[:, 1]).sum())


def _comprimento_suavizado(caminho_yx, s: float) -> float:
    """Comprimento do caminho após suavização por spline (reduz serrilhado)."""
    n = len(caminho_yx)
    if not _TEM_INTERP or n < 4:
        return _comprimento_poligonal(caminho_yx)
    pts = np.asarray(caminho_yx, dtype=np.float64)
    ys, xs = pts[:, 0], pts[:, 1]
    try:
        tck, _ = splprep([xs, ys], s=s, k=min(3, n - 1))
    except Exception:
        return _comprimento_poligonal(caminho_yx)
    u = np.linspace(0.0, 1.0, max(n * 4, 50))
    xs_s, ys_s = splev(u, tck)
    difs = np.diff(np.column_stack([ys_s, xs_s]), axis=0)
    return float(np.hypot(difs[:, 0], difs[:, 1]).sum())


def comprimento_caminho_px(caminho_yx, config) -> float:
    """Comprimento do caminho em pixels, com ou sem suavização (config.medicao)."""
    med = config.medicao
    if getattr(med, "suavizar_caminho", False):
        return _comprimento_suavizado(caminho_yx, float(med.suavizacao_spline))
    return _comprimento_poligonal(caminho_yx)


def medir_segmentos(esqueleto, topo, colo, ponta, escala_mm_px, config) -> ResultadoMedicao:
    """Mede os 2 segmentos da plântula em mm seguindo o caminho real do esqueleto.

    `esqueleto` é a imagem 0/255 (1px) da plântula; `topo`, `colo` e `ponta` são
    pontos (y, x) sobre o esqueleto (Etapa 5); `escala_mm_px` vem da calibração.
    """
    esq_bool = esqueleto > 0
    grafo, pixels, idx_map = construir_grafo(esq_bool)

    caminho1, _ = caminho_entre(grafo, pixels, idx_map, topo, colo)
    caminho2, _ = caminho_entre(grafo, pixels, idx_map, colo, ponta)

    seg1_px = comprimento_caminho_px(caminho1, config)
    seg2_px = comprimento_caminho_px(caminho2, config)

    seg1_mm = seg1_px * escala_mm_px
    seg2_mm = seg2_px * escala_mm_px

    return ResultadoMedicao(
        segmento1_mm=seg1_mm,
        segmento2_mm=seg2_mm,
        total_mm=seg1_mm + seg2_mm,
        caminho1=caminho1,
        caminho2=caminho2,
    )
