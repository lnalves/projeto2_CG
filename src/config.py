"""Parâmetros centralizados do pipeline de medição de plântulas.

A medição é semiautomática por live-wire (ADR 0001): o usuário clica topo e
ponta e o comprimento é o caminho de custo mínimo entre eles. Os parâmetros do
custo (black-hat), do colo e do snap da semente ficam em `ConfigLivewire`.
"""

from dataclasses import dataclass, field


@dataclass
class ConfigCalibracao:
    # Distância real (mm) entre os dois pontos usados na calibração manual.
    # Ex.: marca de 1 cm a 10 cm na régua = 90 mm.
    distancia_conhecida_mm: float = 90.0
    # Confiança mínima da detecção automática dos ticks para dispensar o clique.
    confianca_minima_auto: float = 0.6
    # Distância real (mm) entre dois ticks consecutivos da régua (menor divisão).
    espacamento_tick_mm: float = 1.0
    # ROI da régua para detecção automática: (x, y, w, h). None = sem ROI.
    regua_roi: tuple | None = None
    # Largura máxima (px) das janelas interativas; imagens maiores são reduzidas
    # só para exibição (os cliques voltam às coordenadas originais).
    largura_max_exibicao: int = 1200


@dataclass
class ConfigPreprocessamento:
    clahe_limite_clip: float = 2.0
    clahe_tamanho_grade: int = 8
    gaussiana_ksize: int = 3  # ímpar


@dataclass
class ConfigMedicao:
    # Suavização do caminho por spline antes de somar comprimentos.
    suavizar_caminho: bool = True
    suavizacao_spline: float = 2.0


@dataclass
class ConfigLivewire:
    # Medição semiautomática por caminho de custo mínimo (ADR 0001).
    # Custo: realce black-hat → filamento fica barato, papel caro.
    blackhat_kernel: int = 21        # ímpar; deve ser > largura do filamento
    custo_gamma: float = 3.0         # afia o contraste do custo
    margem_bbox_px: int = 60         # margem do recorte onde o caminho é roteado
    # Colo: detectado pela espessura de uma máscara de filamento (black-hat).
    limiar_blackhat: int = 10        # limiar da máscara de espessura (0 = Otsu)
    # Snap do topo: detecção da semente escura para "grudar" o clique do topo.
    limiar_semente: int = 80         # luminância abaixo disto = semente
    semente_area_min: int = 50       # área mín. (px) para contar como semente
    snap_raio_px: int = 45           # raio (px) para grudar o topo na semente


@dataclass
class ConfigRenderizacao:
    cor_seg1_bgr: tuple = (0, 255, 0)      # hipocótilo (topo → colo)
    cor_seg2_bgr: tuple = (0, 128, 255)    # raiz (colo → ponta)
    raio_ponto: int = 5
    escala_fonte: float = 0.6
    espessura: int = 2


@dataclass
class Config:
    calibracao: ConfigCalibracao = field(default_factory=ConfigCalibracao)
    preprocessamento: ConfigPreprocessamento = field(default_factory=ConfigPreprocessamento)
    medicao: ConfigMedicao = field(default_factory=ConfigMedicao)
    livewire: ConfigLivewire = field(default_factory=ConfigLivewire)
    renderizacao: ConfigRenderizacao = field(default_factory=ConfigRenderizacao)

    # Salvar imagens intermediárias em output/debug/.
    debug: bool = False


def config_padrao() -> Config:
    return Config()
