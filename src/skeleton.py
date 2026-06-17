"""
Esqueletização, transformada de distância e 3 pontos.
Esqueleto 1px, espessura local, pruning, endpoints/bifurcações.
Identificação de topo, colo (estrangulamento) e ponta da raiz.
"""

from __future__ import annotations


def skeletonize_plant(mask, config):
    """Retorna esqueleto + mapa de distância."""

def detect_points(skeleton, dist_map, seed_mask, config):
    """Retorna (topo, colo, ponta)."""
