"""Calibração da escala (mm por pixel) a partir da régua na cena.

Dois caminhos:
  1. Automático: dada uma ROI da régua, detecta os ticks periódicos e estima
     o espaçamento médio em pixels. Se a regularidade for suficiente
     (confiança >= confianca_minima_auto), dispensa o clique.
  2. Manual (fallback): o usuário clica em dois pontos de distância real
     conhecida (distancia_conhecida_mm) e a escala sai da razão.

A função pública `calibrar` devolve a escala em **mm/px** (multiplicar um
comprimento em pixels por essa escala dá o comprimento em milímetros).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

try:
    from scipy.signal import find_peaks
    _TEM_SCIPY = True
except ImportError:
    _TEM_SCIPY = False


@dataclass
class ResultadoCalibracao:
    """Resultado detalhado da calibração."""

    mm_por_px: float
    metodo: str          # "auto" ou "manual"
    confianca: float     # 0..1
    px_por_unidade: float


def calibrar(imagem, config, regua_roi=None) -> float:
    """Retorna a escala mm/px.

    Tenta a detecção automática quando há uma ROI da régua disponível
    (argumento `regua_roi` ou `config.calibracao.regua_roi`). Se a confiança
    ficar abaixo de `confianca_minima_auto`, cai para a calibração manual por
    clique. Levanta `RuntimeError` se nenhum método produzir escala válida.
    """
    resultado = calibrar_detalhado(imagem, config, regua_roi=regua_roi)
    return resultado.mm_por_px


def calibrar_detalhado(imagem, config, regua_roi=None) -> ResultadoCalibracao:
    cal = config.calibracao
    roi = regua_roi if regua_roi is not None else cal.regua_roi

    if roi is not None:
        auto = _calibrar_auto(imagem, roi, cal)
        if auto is not None and auto.confianca >= cal.confianca_minima_auto:
            return auto

    manual = _calibrar_manual(imagem, cal)
    if manual is None:
        raise RuntimeError(
            "Calibração falhou: detecção automática insuficiente e nenhum "
            "ponto foi marcado manualmente."
        )
    return manual


def _calibrar_auto(imagem, roi, cal) -> ResultadoCalibracao | None:
    """Estima a escala pelos ticks periódicos da régua dentro da ROI.

    Retorna None se não houver ticks regulares suficientes.
    """
    x, y, w, h = (int(v) for v in roi)
    recorte = imagem[y : y + h, x : x + w]
    if recorte.size == 0:
        return None

    cinza = recorte if recorte.ndim == 2 else cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)

    # Eixo de medição = lado mais comprido da ROI. Projetamos no eixo curto.
    eixo_medicao_x = w >= h
    perfil = _perfil_ticks(cinza, eixo_medicao_x)
    if perfil is None or perfil.size < 4:
        return None

    espacamento_px, confianca = _estimar_espacamento(perfil)
    if espacamento_px is None or espacamento_px <= 0:
        return None

    mm_por_px = cal.espacamento_tick_mm / espacamento_px
    return ResultadoCalibracao(
        mm_por_px=mm_por_px,
        metodo="auto",
        confianca=confianca,
        px_por_unidade=espacamento_px,
    )


def _perfil_ticks(cinza, eixo_medicao_x) -> np.ndarray | None:
    """Perfil 1D ao longo do eixo de medição, realçando os ticks escuros."""
    # Realce e binarização: ticks costumam ser mais escuros que o corpo da régua.
    suave = cv2.GaussianBlur(cinza, (3, 3), 0)
    binaria = cv2.adaptiveThreshold(
        suave, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 7
    )
    # Soma ao longo do eixo curto → densidade de pixels de tick por coluna/linha.
    eixo = 0 if eixo_medicao_x else 1
    perfil = binaria.sum(axis=eixo).astype(np.float64)
    if perfil.max() <= 0:
        return None
    return perfil / perfil.max()


def _estimar_espacamento(perfil) -> tuple[float | None, float]:
    """Espaçamento mediano entre picos do perfil + confiança (0..1).

    A confiança combina a quantidade de ticks encontrados com a regularidade
    do espaçamento (baixa variação relativa => alta confiança).
    """
    picos = _encontrar_picos(perfil)
    if picos.size < 3:
        return None, 0.0

    difs = np.diff(picos).astype(np.float64)
    difs = difs[difs > 0]
    if difs.size < 2:
        return None, 0.0

    mediana = float(np.median(difs))
    if mediana <= 0:
        return None, 0.0

    # Coeficiente de variação robusto → regularidade.
    cv = float(np.std(difs) / mediana)
    regularidade = max(0.0, 1.0 - cv)

    contagem = min(1.0, difs.size / 10.0)
    confianca = regularidade * contagem
    return mediana, confianca


def _encontrar_picos(perfil) -> np.ndarray:
    if not _TEM_SCIPY:
        raise RuntimeError(
            "SciPy não está instalado: necessário para a detecção automática "
            "dos ticks. Instale com `pip install scipy` ou use a calibração manual."
        )
    # Distância mínima evita pegar o mesmo tick duas vezes; altura corta ruído.
    picos, _ = find_peaks(perfil, height=0.3, distance=3)
    return picos


def _calibrar_manual(imagem, cal) -> ResultadoCalibracao | None:
    """Coleta dois cliques e converte a distância conhecida em mm/px."""
    pontos = _dois_cliques(imagem, cal)
    if pontos is None or len(pontos) < 2:
        return None

    (x0, y0), (x1, y1) = pontos[0], pontos[1]
    dist_px = float(np.hypot(x1 - x0, y1 - y0))
    if dist_px <= 0:
        return None

    mm_por_px = cal.distancia_conhecida_mm / dist_px
    return ResultadoCalibracao(
        mm_por_px=mm_por_px,
        metodo="manual",
        confianca=1.0,
        px_por_unidade=dist_px,
    )


def _dois_cliques(imagem, cal) -> list[tuple[int, int]] | None:
    exibicao, escala = _ajustar_exibicao(imagem, cal.largura_max_exibicao)
    janela = "Calibracao: clique 2 pontos (dist. conhecida) | ENTER ok, ESC cancela"
    cliques: list[tuple[int, int]] = []

    def ao_clicar(evento, mx, my, flags, param):
        if evento == cv2.EVENT_LBUTTONDOWN and len(cliques) < 2:
            # Volta para coordenadas da imagem original.
            cliques.append((int(round(mx / escala)), int(round(my / escala))))

    cv2.namedWindow(janela, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(janela, ao_clicar)
    try:
        while True:
            tela = exibicao.copy()
            for i, (px, py) in enumerate(cliques):
                dp = (int(round(px * escala)), int(round(py * escala)))
                cv2.circle(tela, dp, 5, (0, 0, 255), -1)
                cv2.putText(
                    tela, str(i + 1), (dp[0] + 8, dp[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                )
            if len(cliques) == 2:
                p0 = (int(round(cliques[0][0] * escala)), int(round(cliques[0][1] * escala)))
                p1 = (int(round(cliques[1][0] * escala)), int(round(cliques[1][1] * escala)))
                cv2.line(tela, p0, p1, (0, 0, 255), 2)

            cv2.imshow(janela, tela)
            tecla = cv2.waitKey(20) & 0xFF
            if tecla == 27:  # ESC
                cliques = []
                break
            if tecla in (13, 10) and len(cliques) == 2:  # ENTER
                break
    finally:
        cv2.destroyWindow(janela)

    return cliques if len(cliques) == 2 else None


def _ajustar_exibicao(imagem, largura_max):
    """Reduz a imagem para caber na largura máxima de exibição."""
    h, w = imagem.shape[:2]
    if w <= largura_max:
        return imagem, 1.0
    escala = largura_max / float(w)
    exibicao = cv2.resize(
        imagem, (largura_max, int(round(h * escala))), interpolation=cv2.INTER_AREA
    )
    return exibicao, escala
