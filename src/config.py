"""Parâmetros centralizados do pipeline de medição de plântulas.

Mantenha aqui tudo que é ajustável por imagem/lote, para não espalhar
"números mágicos" pelos módulos. As trackbars do OpenCV (Etapa 3) usam
estes valores como ponto de partida.
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
    # Réguas comuns têm divisões de 1 mm.
    espacamento_tick_mm: float = 1.0
    # ROI da régua para detecção automática: (x, y, w, h). None = sem ROI.
    regua_roi: tuple | None = None
    # Largura máxima (px) da janela de clique manual; imagens maiores são
    # reduzidas só para exibição (os cliques voltam às coordenadas originais).
    largura_max_exibicao: int = 1200


@dataclass
class ConfigPreprocessamento:
    clahe_limite_clip: float = 2.0
    clahe_tamanho_grade: int = 8
    gaussiana_ksize: int = 3  # ímpar


@dataclass
class ConfigSegmentacao:
    # Limiar da semente escura (parte superior, escura/amarelada).
    limiar_semente: int = 80
    # Threshold adaptativo do filamento.
    bloco_adaptativo: int = 31  # ímpar
    c_adaptativo: int = 5
    # Morfologia.
    kernel_morfologico: int = 3
    iter_abertura: int = 1
    iter_fechamento: int = 2
    # Área mínima (px) de um componente para contar como plântula.
    area_minima_px: int = 200


@dataclass
class ConfigEsqueleto:
    # Comprimento mínimo (px) de um ramo para NÃO ser podado (pruning).
    comprimento_min_ramo_px: int = 15


@dataclass
class ConfigMedicao:
    # Suavização do caminho por spline antes de somar comprimentos.
    suavizar_caminho: bool = True
    suavizacao_spline: float = 2.0


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
    segmentacao: ConfigSegmentacao = field(default_factory=ConfigSegmentacao)
    esqueleto: ConfigEsqueleto = field(default_factory=ConfigEsqueleto)
    medicao: ConfigMedicao = field(default_factory=ConfigMedicao)
    renderizacao: ConfigRenderizacao = field(default_factory=ConfigRenderizacao)

    # Salvar imagens intermediárias em output/debug/.
    debug: bool = False


def config_padrao() -> Config:
    return Config()
