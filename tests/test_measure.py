"""Testes do módulo measure (medição de comprimento)."""
from __future__ import annotations

import numpy as np
import pytest

from measure import (
    ResultadoMedicao,
    _comprimento_poligonal,
    _comprimento_suavizado,
    comprimento_caminho_px,
    medir_caminho,
)


class TestComprimentoPoligonal:
    def test_caminho_reto(self):
        caminho = [(0, 0), (0, 10)]
        assert _comprimento_poligonal(caminho) == pytest.approx(10.0)

    def test_caminho_diagonal(self):
        caminho = [(0, 0), (10, 10)]
        assert _comprimento_poligonal(caminho) == pytest.approx(np.hypot(10, 10))

    def test_caminho_vazio(self):
        assert _comprimento_poligonal([]) == 0.0

    def test_ponto_unico(self):
        assert _comprimento_poligonal([(5, 5)]) == 0.0

    def test_caminho_3_pontos(self):
        caminho = [(0, 0), (0, 5), (3, 5)]
        esperado = 5.0 + 3.0  # 5 vertical + 3 horizontal
        assert _comprimento_poligonal(caminho) == pytest.approx(esperado)

    def test_caminho_curvo(self, caminho_curvo):
        comp = _comprimento_poligonal(caminho_curvo)
        assert comp > 0
        # O comprimento deve ser < soma de dist retas (curva é sinuosa)
        pts = np.asarray(caminho_curvo, dtype=np.float64)
        reta = float(np.hypot(pts[-1, 0] - pts[0, 0], pts[-1, 1] - pts[0, 1]))
        assert comp > reta


class TestComprimentoSuavizado:
    def test_fallback_poucos_pontos(self):
        caminho = [(0, 0), (0, 5)]
        assert _comprimento_suavizado(caminho, 1.0) == _comprimento_poligonal(caminho)

    def test_suavizado_menor_que_poligonal(self, caminho_curvo):
        pol = _comprimento_poligonal(caminho_curvo)
        suav = _comprimento_suavizado(caminho_curvo, 2.0)
        # Suavizado deve ser <= poligonal (reduz serrilhado)
        assert suav <= pol * 1.01  # tolerância 1% para diferenças núm.


class TestComprimentoCaminhoPx:
    def test_sem_suavizacao(self, config, caminho_curvo):
        config.medicao.suavizar_caminho = False
        comp = comprimento_caminho_px(caminho_curvo, config)
        assert comp == pytest.approx(_comprimento_poligonal(caminho_curvo))

    def test_com_suavizacao(self, config, caminho_curvo):
        config.medicao.suavizar_caminho = True
        comp = comprimento_caminho_px(caminho_curvo, config)
        assert comp > 0


class TestMedirCaminho:
    def test_medicao_basica(self, config, caminho_curvo):
        n = len(caminho_curvo)
        idx_colo = n // 2
        escala = 0.1  # 0.1 mm/px
        resultado = medir_caminho(caminho_curvo, idx_colo, escala, config)

        assert isinstance(resultado, ResultadoMedicao)
        assert resultado.segmento1_mm > 0
        assert resultado.segmento2_mm > 0
        assert resultado.total_mm == pytest.approx(resultado.segmento1_mm + resultado.segmento2_mm)
        assert len(resultado.caminho1) == idx_colo + 1
        assert len(resultado.caminho2) == n - idx_colo

    def test_colo_no_inicio(self, config, caminho_curvo):
        escala = 0.1
        resultado = medir_caminho(caminho_curvo, 0, escala, config)
        assert resultado.segmento1_mm >= 0
        assert resultado.total_mm > 0

    def test_colo_no_fim(self, config, caminho_curvo):
        n = len(caminho_curvo)
        escala = 0.1
        resultado = medir_caminho(caminho_curvo, n - 1, escala, config)
        assert resultado.segmento2_mm >= 0
        assert resultado.total_mm > 0

    def test_conversao_mm(self, config, caminho_curvo):
        """Verifica que escala_mm_px é aplicada corretamente."""
        n = len(caminho_curvo)
        idx = n // 2

        escala = 0.5  # 0.5 mm/px
        r1 = medir_caminho(caminho_curvo, idx, escala, config)

        escala2 = 1.0  # 1.0 mm/px
        r2 = medir_caminho(caminho_curvo, idx, escala2, config)

        assert r1.total_mm == pytest.approx(r2.total_mm * 0.5)
