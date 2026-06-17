"""
Grafo do esqueleto (orto=1, diag=√2), Dijkstra entre os pontos,
suavização opcional por spline, conversão px → mm.
"""

from __future__ import annotations


def measure_segments(skeleton, topo, colo, ponta, scale_mm_px, config):
    """Retorna (segmento1_mm, segmento2_mm, total_mm, caminhos)."""

