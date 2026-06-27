"""Etapas 4 e 5 — Esqueletização, transformada de distância e 3 pontos.

Esqueleto 1px, espessura local, pruning, endpoints/bifurcações.
Identificação de topo, colo (estrangulamento) e ponta da raiz.
"""

from __future__ import annotations


def esqueletizar_plantula(mascara, config):
    """Retorna esqueleto + mapa de distância. Implementação na Etapa 4."""
    raise NotImplementedError("Esqueletização será implementada na Etapa 4.")


def detectar_pontos(esqueleto, mapa_distancia, mascara_semente, config):
    """Retorna (topo, colo, ponta). Implementação na Etapa 5."""
    raise NotImplementedError("Detecção dos pontos será implementada na Etapa 5.")
