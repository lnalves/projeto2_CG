"""
Detecção automática dos ticks da régua com fallback manual por clique.
"""

from __future__ import annotations


def calibrate(image, config) -> float:
    """Retorna a escala mm/px."""
