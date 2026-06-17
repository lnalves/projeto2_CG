"""Parâmetros centralizados do pipeline de medição de plântulas."""

from dataclasses import dataclass, field


@dataclass
class CalibrationConfig:
    # Distância real (mm) entre os dois pontos usados na calibração manual.
    # Ex.: marca de 1 cm a 10 cm na régua = 90 mm.
    known_distance_mm: float = 90.0
    # Confiança mínima da detecção automática dos ticks para dispensar o clique.
    min_auto_confidence: float = 0.6
    # Distância real (mm) entre dois ticks consecutivos da régua (menor divisão).
    # Réguas comuns têm divisões de 1 mm.
    tick_spacing_mm: float = 1.0
    # ROI da régua para detecção automática: (x, y, w, h). None = sem ROI.
    ruler_roi: tuple | None = None
    # Largura máxima (px) da janela de clique manual; imagens maiores são
    # reduzidas só para exibição (os cliques voltam às coordenadas originais).
    max_display_width: int = 1200


@dataclass
class PreprocessConfig:
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: int = 8
    gaussian_ksize: int = 3  # ímpar


@dataclass
class SegmentationConfig:
    # Threshold da semente escura (parte superior, escura/amarelada).
    seed_thresh: int = 80
    # Threshold adaptativo do filamento.
    adaptive_block_size: int = 31  # ímpar
    adaptive_c: int = 5
    # Morfologia.
    morph_kernel: int = 3
    morph_open_iter: int = 1
    morph_close_iter: int = 2
    # Área mínima (px) de um componente para contar como plântula.
    min_area_px: int = 200


@dataclass
class SkeletonConfig:
    # Comprimento mínimo (px) de um ramo para NÃO ser podado (pruning).
    min_branch_len_px: int = 15


@dataclass
class MeasureConfig:
    # Suavização do caminho por spline antes de somar comprimentos.
    smooth_path: bool = True
    spline_smoothing: float = 2.0


@dataclass
class RenderConfig:
    seg1_color_bgr: tuple = (0, 255, 0)      # hipocótilo (topo → colo)
    seg2_color_bgr: tuple = (0, 128, 255)    # raiz (colo → ponta)
    point_radius: int = 5
    font_scale: float = 0.6
    thickness: int = 2


@dataclass
class Config:
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    skeleton: SkeletonConfig = field(default_factory=SkeletonConfig)
    measure: MeasureConfig = field(default_factory=MeasureConfig)
    render: RenderConfig = field(default_factory=RenderConfig)

    # Salvar imagens intermediárias em output/debug/.
    debug: bool = False


def default_config() -> Config:
    return Config()
