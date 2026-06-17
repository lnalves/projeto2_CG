"""
Pré-processamento (cinza/HSV/Lab, CLAHE, blur).
"""

from __future__ import annotations


def preprocess(image, config):
    """Retorna canais úteis para a segmentação."""

