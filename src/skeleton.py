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


def _centroide_semente(mascara_semente):
    """Centróide (y, x) da semente, ou None se não houver máscara/pixels."""
    if mascara_semente is None:
        return None
    pontos = np.argwhere(mascara_semente > 0)
    if pontos.size == 0:
        return None
    return pontos.mean(axis=0)  # (y, x)


def _escolher_topo(endpoints, mascara_semente):
    """Topo = endpoint mais próximo da semente; sem semente, o mais alto (menor y)."""
    centro = _centroide_semente(mascara_semente)
    if centro is not None:
        return min(
            endpoints,
            key=lambda p: (p[0] - centro[0]) ** 2 + (p[1] - centro[1]) ** 2,
        )
    return min(endpoints, key=lambda p: p[0])  # menor y = mais alto na imagem


def _detectar_colo(caminho, mapa_distancia):
    """Colo = ponto de maior afinamento (queda da espessura) no interior do caminho.

    O hipocótilo é mais grosso e a raiz mais fina; o colo é a transição. Usamos
    a maior queda da espessura suavizada ao longo do caminho, ignorando as
    extremidades (onde a espessura naturalmente cai a zero).
    """
    n = len(caminho)
    if n < 5:
        return caminho[n // 2]

    espessura = np.array(
        [mapa_distancia[y, x] for (y, x) in caminho], dtype=np.float64
    )
    # Suavização por média móvel.
    jan = max(3, n // 15)
    nucleo = np.ones(jan) / jan
    suave = np.convolve(espessura, nucleo, mode="same")

    # Restringe ao interior (15%–85%) para evitar as pontas.
    ini, fim = int(0.15 * n), int(0.85 * n)
    if fim - ini < 2:
        return caminho[n // 2]

    # Maior queda (gradiente mais negativo) marca o estrangulamento.
    grad = np.diff(suave)
    idx_rel = int(np.argmin(grad[ini:fim]))
    idx = ini + idx_rel + 1
    idx = min(max(idx, 0), n - 1)
    return caminho[idx]


def detectar_pontos(esqueleto, mapa_distancia, mascara_semente, config):
    """Identifica (topo, colo, ponta) sobre o esqueleto de uma plântula.

    - topo: extremidade mais próxima da semente (ou a mais alta, sem semente);
    - ponta: extremidade mais distante do topo ao longo do esqueleto (grafo);
    - colo: maior afinamento da espessura no caminho topo → ponta.

    Pontos retornados como (y, x). Levanta ValueError se o esqueleto for vazio.
    """
    from measure import construir_grafo, dijkstra_de, reconstruir_caminho

    esq_bool = esqueleto > 0
    if not esq_bool.any():
        raise ValueError("esqueleto vazio: nada a detectar.")

    grafo, pixels, idx_map = construir_grafo(esq_bool)

    viz = _contar_vizinhos(esq_bool)
    endpoints = [tuple(int(c) for c in p) for p in np.argwhere(viz == 1)]
    if len(endpoints) < 2:
        # Esqueleto sem 2 pontas claras (ex.: laço): usa todos os pixels como
        # candidatos para achar os dois extremos mais afastados.
        endpoints = [tuple(int(c) for c in p) for p in pixels]

    topo = _escolher_topo(endpoints, mascara_semente)
    idx_topo = idx_map[topo]
    dist, pred = dijkstra_de(grafo, idx_topo)

    # Ponta = extremidade alcançável mais distante do topo.
    melhor_idx, melhor_dist = None, -1.0
    for ep in endpoints:
        if ep == topo:
            continue
        i = idx_map[ep]
        if np.isfinite(dist[i]) and dist[i] > melhor_dist:
            melhor_dist, melhor_idx = dist[i], i
    if melhor_idx is None:
        # Nada alcançável além do topo: degenerado.
        return topo, topo, topo

    ponta = (int(pixels[melhor_idx][0]), int(pixels[melhor_idx][1]))
    caminho = reconstruir_caminho(pred, idx_topo, melhor_idx, pixels)
    colo = _detectar_colo(caminho, mapa_distancia)

    return topo, colo, ponta
