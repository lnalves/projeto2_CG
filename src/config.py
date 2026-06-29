"""Parâmetros centralizados do pipeline de medição de plântulas.

A medição é semiautomática por live-wire (ADR 0001): o usuário clica topo e
ponta e o comprimento é o caminho de custo mínimo entre eles.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


class CalibracaoConfig(BaseModel):
    distancia_conhecida_mm: float = Field(default=90.0, gt=0, description="Distância real (mm) entre os 2 pontos da calibração manual.")
    confianca_minima_auto: float = Field(default=0.6, ge=0, le=1, description="Confiança mínima para detecção automática dos ticks.")
    espacamento_tick_mm: float = Field(default=1.0, gt=0, description="Distância real (mm) entre dois ticks consecutivos da régua.")
    largura_max_exibicao: int = Field(default=1200, ge=200, le=4000, description="Largura máxima (px) das janelas interativas.")


class PreprocessamentoConfig(BaseModel):
    clahe_limite_clip: float = Field(default=2.0, ge=0.1, le=10, description="Limite de contraste do CLAHE.")
    clahe_tamanho_grade: int = Field(default=8, ge=2, le=64, description="Tamanho da grade do CLAHE (tileGridSize).")
    gaussiana_ksize: int = Field(default=3, ge=1, le=31, description="Tamanho do kernel Gaussiano (ímpar).")

    @model_validator(mode="after")
    def _validar_ksize_impar(self) -> PreprocessamentoConfig:
        if self.gaussiana_ksize % 2 == 0:
            self.gaussiana_ksize += 1
        return self


class MedicaoConfig(BaseModel):
    suavizar_caminho: bool = Field(default=True, description="Suavizar caminho por spline antes de medir.")
    suavizacao_spline: float = Field(default=2.0, ge=0, description="Parâmetro de suavização s do splprep (0 = interpola exata).")


class LivewireConfig(BaseModel):
    blackhat_kernel: int = Field(default=21, ge=3, le=101, description="Tamanho do kernel do black-hat (ímpar, > largura do filamento).")
    custo_gamma: float = Field(default=3.0, ge=0.1, le=10, description="Gamma da imagem de custo (afia contraste).")
    margem_bbox_px: int = Field(default=60, ge=10, le=500, description="Margem do recorte para roteamento do caminho.")
    bbox_area_max_px: int = Field(default=2_000_000, ge=100_000, le=50_000_000, description="Área máxima do recorte (px²) para roteamento.")
    limiar_blackhat: int = Field(default=10, ge=0, le=255, description="Limiar da máscara de espessura (0 = Otsu).")
    limiar_semente: int = Field(default=80, ge=0, le=255, description="Luminância máxima da semente (abaixo disto = semente).")
    semente_area_min: int = Field(default=50, ge=10, le=10_000, description="Área mínima (px) de um CC para ser semente.")
    snap_raio_px: int = Field(default=45, ge=5, le=200, description="Raio (px) para snap do topo na semente mais próxima.")

    @model_validator(mode="after")
    def _validar_kernel_impar(self) -> LivewireConfig:
        if self.blackhat_kernel % 2 == 0:
            self.blackhat_kernel += 1
        return self


class RenderizacaoConfig(BaseModel):
    cor_seg1_bgr: tuple[int, int, int] = Field(default=(0, 255, 0), description="Cor do segmento 1 (hipocótilo) em BGR.")
    cor_seg2_bgr: tuple[int, int, int] = Field(default=(0, 128, 255), description="Cor do segmento 2 (raiz) em BGR.")
    raio_ponto: int = Field(default=5, ge=1, le=20, description="Raio dos círculos dos pontos (topo/colo/ponta).")
    escala_fonte: float = Field(default=0.6, ge=0.2, le=2, description="Escala da fonte dos rótulos.")
    espessura: int = Field(default=2, ge=1, le=5, description="Espessura das linhas dos caminhos.")


class Config(BaseModel):
    calibracao: CalibracaoConfig = Field(default_factory=CalibracaoConfig)
    preprocessamento: PreprocessamentoConfig = Field(default_factory=PreprocessamentoConfig)
    medicao: MedicaoConfig = Field(default_factory=MedicaoConfig)
    livewire: LivewireConfig = Field(default_factory=LivewireConfig)
    renderizacao: RenderizacaoConfig = Field(default_factory=RenderizacaoConfig)
    debug: bool = Field(default=False, description="Salvar imagens intermediárias em output/debug/.")


def config_padrao() -> Config:
    """Cria Config com valores padrão."""
    return Config()


def carregar_config(caminho: str | Path | None) -> Config:
    """Carrega config de arquivo YAML e faz merge com defaults.

    Args:
        caminho: Caminho do arquivo YAML. Se None, retorna config padrão.

    Returns:
        Config com valores carregados + defaults para campos ausentes.
    """
    if not caminho:
        return config_padrao()

    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo de config não encontrado: {caminho}")

    with open(caminho) as f:
        dados = yaml.safe_load(f)

    if not dados:
        return config_padrao()

    return Config(**dados)
