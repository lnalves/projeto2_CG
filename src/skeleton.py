"""

Etapa 4 (este arquivo, `esqueletizar_plantula`):
  - Reduz a máscara da plântula a um **esqueleto de 1 pixel** (linha central do
    filamento) via `skeletonize` (scikit-image), com fallback para
    `cv2.ximgproc.thinning`.
  - Calcula a **transformada de distância** sobre a máscara → espessura local
    ao longo do filamento (usada na Etapa 5 para achar o colo).
  - **Poda** (pruning) ramos espúrios curtos para isolar o caminho principal.
  - Detecta **endpoints** (1 vizinho) e **bifurcações** (>=3 vizinhos).

Convenção de coordenadas: pontos são tuplas `(y, x)` (linha, coluna) — ordem de
índice NumPy. Ao desenhar com OpenCV, lembre de inverter para `(x, y)`.

"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.ndimage import convolve

try:
    from skimage.morphology import skeletonize as _sk_skeletonize
    _TEM_SKIMAGE = True
except ImportError:
    _TEM_SKIMAGE = False

# Kernel 3x3 sem o centro: convoluído com o esqueleto binário (0/1) dá o número
# de vizinhos de cada pixel do esqueleto.
_KERNEL_VIZINHOS = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)


@dataclass
class ResultadoEsqueleto:
    """Saída da esqueletização de uma máscara (uma plântula ou a máscara toda)."""

    esqueleto: np.ndarray            # uint8 0/255, 1px de largura, já podado
    mapa_distancia: np.ndarray       # float32: espessura local (dist. ao fundo)
    endpoints: list = field(default_factory=list)     # [(y, x), ...] grau 1
    bifurcacoes: list = field(default_factory=list)   # [(y, x), ...] grau >=3


def _para_binaria(mascara: np.ndarray) -> np.ndarray:
    """Converte uma máscara qualquer para booleana (True = primeiro plano)."""
    if mascara.ndim != 2:
        raise ValueError("esqueletização espera máscara 2D (cinza/binária).")
    return mascara > 0


def _esqueletizar(bin_bool: np.ndarray) -> np.ndarray:
    """Esqueleto 1px (bool). Usa scikit-image; cai para ximgproc.thinning."""
    if _TEM_SKIMAGE:
        return _sk_skeletonize(bin_bool)
    fino = cv2.ximgproc.thinning((bin_bool.astype(np.uint8)) * 255)
    return fino > 0


def _contar_vizinhos(esq_bool: np.ndarray) -> np.ndarray:
    """Mapa com o número de vizinhos (8-conectados) de cada pixel do esqueleto."""
    viz = convolve(esq_bool.astype(np.uint8), _KERNEL_VIZINHOS, mode="constant", cval=0)
    return viz * esq_bool  # zera fora do esqueleto


def _endpoints_bifurcacoes(esq_bool: np.ndarray):
    """Retorna (endpoints, bifurcacoes) como listas de (y, x)."""
    viz = _contar_vizinhos(esq_bool)
    endpoints = [tuple(p) for p in np.argwhere(viz == 1)]
    bifurcacoes = [tuple(p) for p in np.argwhere(viz >= 3)]
    return endpoints, bifurcacoes


def _vizinhos_esqueleto(esq_bool, y, x):
    """Coordenadas (y, x) dos vizinhos pertencentes ao esqueleto."""
    h, w = esq_bool.shape
    out = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and esq_bool[ny, nx]:
                out.append((ny, nx))
    return out


def _tracar_ramo(esq_bool, viz, inicio):
    """Percorre um ramo a partir de um endpoint até uma bifurcação/endpoint.

    Devolve a lista de pixels do ramo SEM incluir a bifurcação onde ele termina
    (para não desconectar o resto do esqueleto ao podar).
    """
    caminho = [inicio]
    anterior = None
    atual = inicio
    # Limite de segurança contra ciclos.
    for _ in range(esq_bool.shape[0] * esq_bool.shape[1]):
        vizinhos = [v for v in _vizinhos_esqueleto(esq_bool, *atual) if v != anterior]
        if len(vizinhos) != 1:
            # 0 vizinhos (ramo isolado) ou >=2 (chegou numa bifurcação): para.
            break
        prox = vizinhos[0]
        if viz[prox] >= 3:
            # Próximo é bifurcação: termina o ramo aqui, sem incluí-la.
            break
        caminho.append(prox)
        anterior, atual = atual, prox
    return caminho


def _podar(esq_bool: np.ndarray, comprimento_min: int) -> np.ndarray:
    """Remove ramos curtos (< comprimento_min) iterativamente até estabilizar."""
    esq = esq_bool.copy()
    if comprimento_min <= 0:
        return esq
    while True:
        viz = _contar_vizinhos(esq)
        endpoints = [tuple(p) for p in np.argwhere(viz == 1)]
        removeu = False
        for ep in endpoints:
            if not esq[ep]:
                continue  # já removido nesta passada
            ramo = _tracar_ramo(esq, viz, ep)
            # Só poda ramos curtos que terminam numa bifurcação (espúrios);
            # ramos curtos isolados (a plântula inteira pequena) são preservados.
            termina_em_bifurcacao = len(ramo) >= 1 and any(
                viz[v] >= 3 for v in _vizinhos_esqueleto(esq, *ramo[-1])
            )
            if len(ramo) < comprimento_min and termina_em_bifurcacao:
                for px in ramo:
                    esq[px] = False
                removeu = True
        if not removeu:
            break
    return esq


def esqueletizar_plantula(mascara, config) -> ResultadoEsqueleto:
    """Esqueletiza uma máscara binária e devolve esqueleto + mapa de distância.

    Aceita a máscara de uma única plântula (ex.: `rotulos == id`) ou a máscara
    completa. Parâmetros de poda vêm de `config.esqueleto`.
    """
    bin_bool = _para_binaria(mascara)

    # Espessura local: distância de cada pixel da máscara ao fundo mais próximo.
    mapa_distancia = cv2.distanceTransform(
        (bin_bool.astype(np.uint8)) * 255, cv2.DIST_L2, 5
    )

    esq_bool = _esqueletizar(bin_bool)
    esq_bool = _podar(esq_bool, int(config.esqueleto.comprimento_min_ramo_px))

    endpoints, bifurcacoes = _endpoints_bifurcacoes(esq_bool)

    return ResultadoEsqueleto(
        esqueleto=(esq_bool.astype(np.uint8)) * 255,
        mapa_distancia=mapa_distancia,
        endpoints=endpoints,
        bifurcacoes=bifurcacoes,
    )


def detectar_pontos(esqueleto, mapa_distancia, mascara_semente, config):
    """Retorna (topo, colo, ponta). Implementação na Etapa 5."""
    raise NotImplementedError("Detecção dos pontos será implementada na Etapa 5.")
