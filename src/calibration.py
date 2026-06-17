"""
Calibração da escala (mm por pixel) a partir da régua na cena.

Dois caminhos:
  1. Automático: dada uma ROI da régua, detecta os ticks periódicos e estima
     o espaçamento médio em pixels. Se a regularidade for suficiente
     (confiança >= min_auto_confidence), dispensa o clique.
  2. Manual (fallback): o usuário clica em dois pontos de distância real
     conhecida (known_distance_mm) e a escala sai da razão.

A função pública `calibrate` devolve a escala em **mm/px** (multiplicar um
comprimento em pixels por essa escala dá o comprimento em milímetros).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

try:
    from scipy.signal import find_peaks
    _HAS_SCIPY = True
except ImportError:  
    _HAS_SCIPY = False


@dataclass
class CalibrationResult:
    """Resultado detalhado da calibração."""

    mm_per_px: float
    method: str    
    confidence: float   
    px_per_unit: float


def calibrate(image, config, ruler_roi=None) -> float:
    """Retorna a escala mm/px.

    Tenta a detecção automática quando há uma ROI da régua disponível
    (argumento `ruler_roi` ou `config.calibration.ruler_roi`). Se a confiança
    ficar abaixo de `min_auto_confidence`, cai para a calibração manual por
    clique. Levanta `RuntimeError` se nenhum método produzir uma escala válida.
    """
    result = calibrate_detailed(image, config, ruler_roi=ruler_roi)
    return result.mm_per_px


def calibrate_detailed(image, config, ruler_roi=None) -> CalibrationResult:
    """Como `calibrate`, porém devolve o `CalibrationResult` completo."""
    cal = config.calibration
    roi = ruler_roi if ruler_roi is not None else cal.ruler_roi

    if roi is not None:
        auto = _auto_calibrate(image, roi, cal)
        if auto is not None and auto.confidence >= cal.min_auto_confidence:
            return auto

    manual = _manual_calibrate(image, cal)
    if manual is None:
        raise RuntimeError(
            "Calibração falhou: detecção automática insuficiente e nenhum "
            "ponto foi marcado manualmente."
        )
    return manual


# --------------------------------------------------------------------------- #
# Detecção automática dos ticks
# --------------------------------------------------------------------------- #
def _auto_calibrate(image, roi, cal) -> CalibrationResult | None:
    """Estima a escala pelos ticks periódicos da régua dentro da ROI.

    Retorna None se não houver ticks regulares suficientes.
    """
    x, y, w, h = (int(v) for v in roi)
    crop = image[y : y + h, x : x + w]
    if crop.size == 0:
        return None

    gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Eixo de medição = lado mais comprido da ROI. Projetamos no eixo curto.
    measure_axis_is_x = w >= h
    profile = _tick_profile(gray, measure_axis_is_x)
    if profile is None or profile.size < 4:
        return None

    spacing_px, confidence = _estimate_spacing(profile)
    if spacing_px is None or spacing_px <= 0:
        return None

    mm_per_px = cal.tick_spacing_mm / spacing_px
    return CalibrationResult(
        mm_per_px=mm_per_px,
        method="auto",
        confidence=confidence,
        px_per_unit=spacing_px,
    )


def _tick_profile(gray, measure_axis_is_x) -> np.ndarray | None:
    """Perfil 1D ao longo do eixo de medição, realçando os ticks escuros."""
    # Realce e binarização: ticks costumam ser mais escuros que o corpo da régua.
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 7
    )
    # Soma ao longo do eixo curto → densidade de pixels de tick por coluna/linha.
    axis = 0 if measure_axis_is_x else 1
    profile = binary.sum(axis=axis).astype(np.float64)
    if profile.max() <= 0:
        return None
    return profile / profile.max()


def _estimate_spacing(profile) -> tuple[float | None, float]:
    """Espaçamento mediano entre picos do perfil + confiança (0..1).

    A confiança combina a quantidade de ticks encontrados com a regularidade
    do espaçamento (baixa variação relativa => alta confiança).
    """
    peaks = _find_peaks(profile)
    if peaks.size < 3:
        return None, 0.0

    diffs = np.diff(peaks).astype(np.float64)
    diffs = diffs[diffs > 0]
    if diffs.size < 2:
        return None, 0.0

    median = float(np.median(diffs))
    if median <= 0:
        return None, 0.0

    # Coeficiente de variação robusto → regularidade.
    cv = float(np.std(diffs) / median)
    regularity = max(0.0, 1.0 - cv)

    count_factor = min(1.0, diffs.size / 10.0)
    confidence = regularity * count_factor
    return median, confidence


def _find_peaks(profile) -> np.ndarray:
    """Picos do perfil (com SciPy se disponível, senão fallback simples)."""
    if _HAS_SCIPY:
        # Distância mínima evita pegar o mesmo tick duas vezes; altura corta ruído.
        peaks, _ = find_peaks(profile, height=0.3, distance=3)
        return peaks

    thr = 0.3
    left = profile[1:-1] > profile[:-2]
    right = profile[1:-1] >= profile[2:]
    above = profile[1:-1] > thr
    return np.where(left & right & above)[0] + 1


def _manual_calibrate(image, cal) -> CalibrationResult | None:
    """Coleta dois cliques e converte a distância conhecida em mm/px."""
    points = _collect_two_clicks(image, cal)
    if points is None or len(points) < 2:
        return None

    (x0, y0), (x1, y1) = points[0], points[1]
    px_dist = float(np.hypot(x1 - x0, y1 - y0))
    if px_dist <= 0:
        return None

    mm_per_px = cal.known_distance_mm / px_dist
    return CalibrationResult(
        mm_per_px=mm_per_px,
        method="manual",
        confidence=1.0,
        px_per_unit=px_dist,
    )


def _collect_two_clicks(image, cal) -> list[tuple[int, int]] | None:
    disp, scale = _fit_for_display(image, cal.max_display_width)
    window = "Calibracao: clique 2 pontos (dist. conhecida) | ENTER ok, ESC cancela"
    clicks: list[tuple[int, int]] = []

    def on_mouse(event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 2:
            # Volta para coordenadas da imagem original.
            clicks.append((int(round(mx / scale)), int(round(my / scale))))

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)
    try:
        while True:
            canvas = disp.copy()
            for i, (px, py) in enumerate(clicks):
                dp = (int(round(px * scale)), int(round(py * scale)))
                cv2.circle(canvas, dp, 5, (0, 0, 255), -1)
                cv2.putText(
                    canvas, str(i + 1), (dp[0] + 8, dp[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                )
            if len(clicks) == 2:
                p0 = (int(round(clicks[0][0] * scale)), int(round(clicks[0][1] * scale)))
                p1 = (int(round(clicks[1][0] * scale)), int(round(clicks[1][1] * scale)))
                cv2.line(canvas, p0, p1, (0, 0, 255), 2)

            cv2.imshow(window, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key == 27:  # ESC
                clicks = []
                break
            if key in (13, 10) and len(clicks) == 2:  # ENTER
                break
    finally:
        cv2.destroyWindow(window)

    return clicks if len(clicks) == 2 else None


def _fit_for_display(image, max_width):
    """Reduz a imagem para caber na largura máxima de exibição.
    """
    h, w = image.shape[:2]
    if w <= max_width:
        return image, 1.0
    scale = max_width / float(w)
    disp = cv2.resize(image, (max_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return disp, scale
