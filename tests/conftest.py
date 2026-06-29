"""Fixtures e utilitários compartilhados entre os testes."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import Config, config_padrao


@pytest.fixture
def config() -> Config:
    """Config padrão limpa para testes."""
    return config_padrao()


@pytest.fixture
def imagem_branca() -> np.ndarray:
    """Imagem branca 200x300 (simula papel branco)."""
    return np.ones((200, 300, 3), dtype=np.uint8) * 255


@pytest.fixture
def imagem_cinza() -> np.ndarray:
    """Imagem 8-bit 200x300 com gradiente suave."""
    return np.tile(np.linspace(50, 200, 300, dtype=np.uint8), (200, 1))


@pytest.fixture
def caminho_curvo() -> list:
    """Caminho sintético de ~30 pontos formando uma curva suave."""
    pts = []
    for i in range(30):
        x = 50 + i * 3
        y = 100 + int(20 * np.sin(i * 0.3))
        pts.append((y, x))
    return pts


@pytest.fixture
def pasta_debug(tmp_path: Path) -> str:
    """Pasta temporária para debug de pré-processamento."""
    p = tmp_path / "debug"
    p.mkdir(parents=True, exist_ok=True)
    return str(p)
