"""
Segmentação + separação das plântulas.
Threshold (semente + filamento), morfologia, componentes conexos.
Trackbars para ajuste ao vivo.
"""

from __future__ import annotations


def segment(channels, config):
    """Retorna máscara binária + rótulos das plântulas."""

