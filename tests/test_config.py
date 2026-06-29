"""Testes do módulo config (carregamento e validação)."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from config import (
    CalibracaoConfig,
    Config,
    LivewireConfig,
    PreprocessamentoConfig,
    carregar_config,
    config_padrao,
)


class TestCalibracaoConfig:
    def test_defaults(self):
        c = CalibracaoConfig()
        assert c.distancia_conhecida_mm == 90.0
        assert c.largura_max_exibicao == 1200

    def test_validacao_distancia(self):
        with pytest.raises(ValidationError):
            CalibracaoConfig(distancia_conhecida_mm=-1)

    def test_validacao_confianca(self):
        with pytest.raises(ValidationError):
            CalibracaoConfig(confianca_minima_auto=1.5)


class TestPreprocessamentoConfig:
    def test_defaults(self):
        p = PreprocessamentoConfig()
        assert p.clahe_limite_clip == 2.0
        assert p.gaussiana_ksize == 3

    def test_ksize_impar(self):
        p = PreprocessamentoConfig(gaussiana_ksize=4)
        assert p.gaussiana_ksize % 2 == 1


class TestLivewireConfig:
    def test_defaults(self):
        lw = LivewireConfig()
        assert lw.blackhat_kernel == 21
        assert lw.bbox_area_max_px == 2_000_000

    def test_kernel_impar(self):
        lw = LivewireConfig(blackhat_kernel=20)
        assert lw.blackhat_kernel % 2 == 1

    def test_validacao_gamma(self):
        with pytest.raises(ValidationError):
            LivewireConfig(custo_gamma=0.0)  # deve ser >= 0.1


class TestConfig:
    def test_config_padrao(self):
        c = config_padrao()
        assert isinstance(c, Config)
        assert isinstance(c.calibracao, CalibracaoConfig)
        assert isinstance(c.livewire, LivewireConfig)
        assert c.debug is False

    def test_config_com_override(self):
        c = Config(
            calibracao=CalibracaoConfig(distancia_conhecida_mm=100),
            livewire=LivewireConfig(blackhat_kernel=31),
            debug=True,
        )
        assert c.calibracao.distancia_conhecida_mm == 100
        assert c.livewire.blackhat_kernel == 31
        assert c.debug is True


class TestCarregarConfig:
    def test_none_retorna_default(self):
        c = carregar_config(None)
        assert isinstance(c, Config)

    def test_arquivo_inexistente(self):
        with pytest.raises(FileNotFoundError):
            carregar_config("/caminho/inexistente.yaml")

    def test_carregar_yaml(self, tmp_path: Path):
        dados = {
            "calibracao": {"distancia_conhecida_mm": 50},
            "livewire": {"blackhat_kernel": 15, "custo_gamma": 2.0},
            "debug": True,
        }
        caminho = tmp_path / "config.yaml"
        with open(caminho, "w") as f:
            yaml.dump(dados, f)

        c = carregar_config(str(caminho))
        assert c.calibracao.distancia_conhecida_mm == 50
        assert c.livewire.blackhat_kernel == 15
        assert c.livewire.custo_gamma == 2.0
        assert c.debug is True
        # Deve ter defaults pros não informados
        assert c.preprocessamento.clahe_limite_clip == 2.0

    def test_yaml_vazio(self, tmp_path: Path):
        caminho = tmp_path / "vazio.yaml"
        with open(caminho, "w") as f:
            f.write("")
        c = carregar_config(str(caminho))
        assert isinstance(c, Config)
        assert c.calibracao.distancia_conhecida_mm == 90.0
