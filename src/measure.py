"""Medição do comprimento ao longo do caminho + conversão de escala.

Na medição semiautomática (live-wire, ADR 0001), o caminho já vem ordenado
(topo→ponta) do roteamento de custo mínimo; aqui ele é partido no colo e cada
trecho é convertido para mm. Opcionalmente o caminho é suavizado por spline
antes de medir, reduzindo o serrilhado (que tende a superestimar o comprimento).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from log import log

try:
    from scipy.interpolate import splev, splprep
    _TEM_INTERP = True
except ImportError:
    _TEM_INTERP = False
    log.warning("SciPy não disponível — spline desativado, usando soma poligonal.")


@dataclass
class ResultadoMedicao:
    """Medidas dos dois segmentos da plântula (em mm) + caminhos percorridos."""

    segmento1_mm: float
    segmento2_mm: float
    total_mm: float
    caminho1: list = field(default_factory=list)  # topo → colo
    caminho2: list = field(default_factory=list)  #  colo → ponta


# --- Comprimento ao longo do caminho ---------------------------------------

def _comprimento_poligonal(caminho_yx) -> float:
    """Soma das distâncias entre pixels consecutivos do caminho."""
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
    except Exception as exc:
        log.warning("Spline falhou (n={}): {} — usando soma poligonal.", n, exc)
        return _comprimento_poligonal(caminho_yx)
    u = np.linspace(0.0, 1.0, max(n * 4, 50))
    xs_s, ys_s = splev(u, tck)
    difs = np.diff(np.column_stack([ys_s, xs_s]), axis=0)
    comp = float(np.hypot(difs[:, 0], difs[:, 1]).sum())
    log.debug("Comprimento: poligonal={:.2f}px → spline={:.2f}px (s={})", _comprimento_poligonal(caminho_yx), comp, s)
    return comp


def comprimento_caminho_px(caminho_yx, config) -> float:
    """Comprimento do caminho em pixels, com ou sem suavização (config.medicao)."""
    med = config.medicao
    if med.suavizar_caminho:
        return _comprimento_suavizado(caminho_yx, float(med.suavizacao_spline))
    return _comprimento_poligonal(caminho_yx)


def medir_caminho(caminho, idx_colo, escala_mm_px, config) -> ResultadoMedicao:
    """Mede um caminho já ordenado (topo→ponta), partido no índice do colo.

    O `caminho` é a lista de pixels (y, x) do caminho de custo mínimo entre os
    2 cliques (live-wire) e `idx_colo` é a posição do colo nessa lista.
    """
    idx = max(1, min(int(idx_colo), len(caminho) - 1)) if len(caminho) > 1 else 0
    caminho1 = caminho[: idx + 1]
    caminho2 = caminho[idx:]

    seg1_mm = comprimento_caminho_px(caminho1, config) * escala_mm_px
    seg2_mm = comprimento_caminho_px(caminho2, config) * escala_mm_px

    if seg1_mm < 0.5 or seg2_mm < 0.5:
        log.warning(
            "Segmento curto: seg1={:.2f}mm, seg2={:.2f}mm. "
            "Verifique a posição do colo.", seg1_mm, seg2_mm,
        )

    return ResultadoMedicao(
        segmento1_mm=seg1_mm,
        segmento2_mm=seg2_mm,
        total_mm=seg1_mm + seg2_mm,
        caminho1=caminho1,
        caminho2=caminho2,
    )
