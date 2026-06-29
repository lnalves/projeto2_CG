"""Testes das funções-núcleo do livewire (testáveis sem GUI)."""
from __future__ import annotations

import numpy as np

from livewire import (
    _blackhat_raw,
    _impar,
    _mascara_blackhat,
    caminho_custo_minimo,
    detectar_sementes,
    imagem_custo,
    indice_colo,
    snap_topo,
)


class TestImpar:
    def test_impar_mantem(self):
        assert _impar(5) == 5

    def test_par_arredonda(self):
        assert _impar(6) == 7

    def test_minimo(self):
        assert _impar(0, minimo=3) == 3
        assert _impar(1, minimo=3) == 3
        assert _impar(2, minimo=3) == 3
        assert _impar(3, minimo=3) == 3

    def test_valores_grandes(self):
        assert _impar(100) == 101
        assert _impar(101) == 101


class TestBlackHatRaw:
    def test_imagem_uniforme(self):
        """Imagem uniforme → black-hat = 0 em toda parte."""
        img = np.ones((50, 50), dtype=np.uint8) * 128
        bh = _blackhat_raw(img, 11)
        assert bh.shape == (50, 50)
        assert bh.dtype == np.float64
        assert bh.max() == 0.0
        assert bh.min() == 0.0

    def test_estrutura_escura(self):
        """Estrutura escura fina sobre fundo claro → realçada."""
        img = np.ones((60, 60), dtype=np.uint8) * 200
        img[20:40, 28:32] = 60  # faixa escura vertical
        bh = _blackhat_raw(img, 11)
        # A faixa escura deve ter valores > 0 no black-hat
        regiao = bh[20:40, 28:32]
        assert regiao.mean() > 0


class TestMascaraBlackhat:
    def test_mascara_binaria(self):
        img = np.ones((50, 50), dtype=np.uint8) * 200
        img[20:30, 23:27] = 50
        masc = _mascara_blackhat(img, 11, 10)
        assert masc.dtype == np.uint8
        assert set(np.unique(masc)).issubset({0, 1})

    def test_limiar_alto_vazio(self):
        img = np.ones((50, 50), dtype=np.uint8) * 200
        img[20:30, 23:27] = 50
        masc = _mascara_blackhat(img, 11, 255)
        assert masc.max() == 0


class TestIndiceColo:
    def test_caminho_curto_fallback(self):
        caminho = [(0, 0), (1, 1), (2, 2)]
        esp = np.ones((5, 5), dtype=np.float64) * 10
        assert indice_colo(caminho, esp) == 1  # n//2

    def test_caminho_com_estrangulamento(self):
        """Cria caminho onde espessura cai no meio."""
        n = 50
        caminho = [(i, 100) for i in range(n)]
        esp = np.ones((n + 10, 110), dtype=np.float64) * 5
        # Espessura alta no início, baixa no fim
        for i in range(n):
            esp[i, 100] = 10 - i * 0.15  # cai linearmente
        idx = indice_colo(caminho, esp)
        # O colo deve estar mais perto do fim (queda acumulada)
        assert idx > 0
        assert idx < n - 1

    def test_colo_no_meio(self):
        """Queda brusca no meio."""
        n = 40
        caminho = [(i, 50) for i in range(n)]
        esp = np.ones((n + 10, 60), dtype=np.float64) * 8
        for i in range(n):
            esp[i, 50] = 8 if i < 20 else 2  # queda abrupta
        idx = indice_colo(caminho, esp)
        assert 15 <= idx <= 25  # ~meio


class MockCanais:
    """Simula canais de pré-processamento para testes."""
    def __init__(self, h=100, w=100):
        self.lab_l = np.ones((h, w), dtype=np.uint8) * 180
        self.valor = np.ones((h, w), dtype=np.uint8) * 200


class TestImagemCusto:
    def test_custo_forma(self, config):
        canais = MockCanais()
        custo = imagem_custo(canais, config)
        assert custo.shape == (100, 100)
        assert custo.dtype == np.float64
        assert custo.min() >= 1e-3  # piso
        assert custo.max() <= 1.0 + 1e-3

    def test_custo_nao_nulo(self, config):
        """Custo tem piso 1e-3 para evitar degeneração do Dijkstra."""
        canais = MockCanais()
        custo = imagem_custo(canais, config)
        assert np.all(custo >= 1e-3)

    def test_custo_baixo_no_filamento(self, config):
        """Filamento (estrutura fina escura sobre fundo claro) → custo baixo."""
        canais = MockCanais(h=50, w=50)
        canais.lab_l[:] = 200  # fundo claro
        canais.lab_l[20:30, 23:27] = 60  # filamento escuro fino
        custo = imagem_custo(canais, config)
        regiao_filamento = custo[20:30, 23:27]
        assert regiao_filamento.mean() < 0.5


class TestCaminhoCustoMinimo:
    def test_caminho_reto(self):
        custo = np.ones((50, 50), dtype=np.float64) * 0.5
        origem, destino = (5, 5), (5, 45)
        caminho = caminho_custo_minimo(custo, origem, destino, 10)
        assert caminho is not None
        assert caminho[0] == origem
        assert caminho[-1] == destino

    def test_caminho_com_canal(self):
        """Canal de custo baixo no meio → caminho deve segui-lo."""
        custo = np.ones((50, 80), dtype=np.float64) * 1.0
        custo[20:30, :] = 0.01  # canal de baixo custo
        origem, destino = (25, 5), (25, 75)
        caminho = caminho_custo_minimo(custo, origem, destino, 15)
        assert caminho is not None
        # O caminho deve ficar no canal de baixo custo
        ys = [p[0] for p in caminho]
        assert min(ys) >= 18 or max(ys) <= 32

    def test_area_excedida(self):
        """Recorte grande demais → None."""
        custo = np.ones((2000, 2000), dtype=np.float64)
        origem, destino = (100, 100), (1800, 1800)
        caminho = caminho_custo_minimo(custo, origem, destino, 500, area_max=100_000)
        assert caminho is None


class TestDetectarSementes:
    def test_sem_sementes(self, config):
        """Imagem clara sem sementes escuras."""
        canais = MockCanais(h=50, w=50)
        canais.valor[:] = 200  # claro → sem sementes
        sementes = detectar_sementes(canais, config)
        assert sementes == []

    def test_uma_semente(self, config):
        """Uma região escura deve ser detectada."""
        canais = MockCanais(h=60, w=60)
        canais.valor[:] = 200
        canais.valor[5:15, 10:20] = 30  # semente escura
        sementes = detectar_sementes(canais, config)
        assert len(sementes) == 1
        assert sementes[0].shape[1] == 2  # (N, 2)


class TestSnapTopo:
    def test_snap_na_semente(self):
        clique = (12, 15)
        semente = np.array([
            [5, 10], [5, 11], [6, 10], [6, 11],
            [7, 10], [7, 11],
        ])
        snapped = snap_topo(clique, [semente], raio=20)
        # y max = 7, xs = [10, 11] → media = 10.5 → int = 10
        assert snapped == (7, 10)

    def test_snap_fora_do_raio(self):
        clique = (50, 50)
        semente = np.array([[5, 5], [5, 6]])
        snapped = snap_topo(clique, [semente], raio=10)
        assert snapped == (50, 50)  # clique original

    def test_sem_sementes(self):
        snapped = snap_topo((10, 10), [], raio=20)
        assert snapped == (10, 10)

    def test_multiplas_sementes(self):
        clique = (15, 15)
        s1 = np.array([[5, 5], [5, 6]])
        s2 = np.array([[14, 14], [14, 15], [15, 14], [15, 15]])
        snapped = snap_topo(clique, [s1, s2], raio=10)
        assert snapped != clique  # deve grudar na s2
        assert snapped[0] == 15  # ymax da s2
