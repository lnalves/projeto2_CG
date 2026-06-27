"""Etapa 2 — Pré-processamento e realce.

O desafio do trabalho é que a plântula é esbranquiçada sobre papel branco.
Para a segmentação (Etapa 3) ter material bom para trabalhar, geramos vários
canais que realçam estruturas diferentes:

  - `cinza`     : tons de cinza com CLAHE (contraste local) → bom para o
                  filamento claro do hipocótilo/raiz.
  - `saturacao` : canal S do HSV → a semente/cotilédones (amarelada) e partes
                  pigmentadas têm saturação maior que o papel branco.
  - `valor`     : canal V do HSV (luminância) → separa o escuro do claro.
  - `lab_l`     : canal L do Lab com CLAHE → luminância perceptual realçada.
  - `lab_b`     : canal b do Lab (eixo azul–amarelo) → destaca a semente
                  amarelada do fundo neutro.
  - `bgr_suavizada`: imagem BGR suavizada (base das conversões e p/ visualização).

A suavização é aplicada na BGR antes das conversões, reduzindo ruído antes de
segmentar.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ResultadoPreprocessamento:
    """Canais úteis para a segmentação (Etapa 3)."""

    cinza: np.ndarray          # cinza + CLAHE
    saturacao: np.ndarray      # HSV: S
    valor: np.ndarray          # HSV: V
    lab_l: np.ndarray          # Lab: L + CLAHE
    lab_b: np.ndarray          # Lab: b (azul–amarelo)
    bgr_suavizada: np.ndarray  # BGR suavizada (base/visualização)


def _impar(n: int) -> int:
    """Garante kernel ímpar (>=1), como o GaussianBlur exige."""
    n = max(1, int(n))
    return n if n % 2 == 1 else n + 1


def _para_bgr(imagem: np.ndarray) -> np.ndarray:
    """Aceita imagem em cinza ou BGR; devolve sempre BGR de 3 canais."""
    if imagem.ndim == 2:
        return cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR)
    return imagem


def preprocessar(imagem, config, dir_debug: str | None = None) -> ResultadoPreprocessamento:
    """Gera os canais de realce a partir da imagem de entrada (BGR ou cinza).

    Parâmetros vêm de `config.preprocessamento` (clip do CLAHE, grade e tamanho
    do blur). Se `config.debug` e `dir_debug` forem informados, salva cada canal
    em PNG dentro de `dir_debug` para conferência visual.
    """
    if imagem is None:
        raise ValueError("preprocessar recebeu imagem vazia (None).")

    pre = config.preprocessamento
    bgr = _para_bgr(imagem)

    # 1) Suavização leve na BGR (base de todas as conversões).
    ksize = _impar(pre.gaussiana_ksize)
    suavizada = cv2.GaussianBlur(bgr, (ksize, ksize), 0)

    # CLAHE reutilizado nos canais de luminância.
    clahe = cv2.createCLAHE(
        clipLimit=pre.clahe_limite_clip,
        tileGridSize=(pre.clahe_tamanho_grade, pre.clahe_tamanho_grade),
    )

    # 2) Cinza + CLAHE.
    cinza = cv2.cvtColor(suavizada, cv2.COLOR_BGR2GRAY)
    cinza_eq = clahe.apply(cinza)

    # 3) HSV → S e V.
    hsv = cv2.cvtColor(suavizada, cv2.COLOR_BGR2HSV)
    _, saturacao, valor = cv2.split(hsv)

    # 4) Lab → L (com CLAHE) e b.
    lab = cv2.cvtColor(suavizada, cv2.COLOR_BGR2LAB)
    lab_l, _, lab_b = cv2.split(lab)
    lab_l_eq = clahe.apply(lab_l)

    resultado = ResultadoPreprocessamento(
        cinza=cinza_eq,
        saturacao=saturacao,
        valor=valor,
        lab_l=lab_l_eq,
        lab_b=lab_b,
        bgr_suavizada=suavizada,
    )

    if getattr(config, "debug", False) and dir_debug:
        salvar_debug(resultado, dir_debug)

    return resultado


def salvar_debug(resultado: ResultadoPreprocessamento, dir_debug: str) -> None:
    """Salva cada canal em PNG dentro de `dir_debug` (cria a pasta se preciso)."""
    os.makedirs(dir_debug, exist_ok=True)
    canais = {
        "preprocess_cinza.png": resultado.cinza,
        "preprocess_saturacao.png": resultado.saturacao,
        "preprocess_valor.png": resultado.valor,
        "preprocess_lab_l.png": resultado.lab_l,
        "preprocess_lab_b.png": resultado.lab_b,
        "preprocess_suavizada.png": resultado.bgr_suavizada,
    }
    for nome, img in canais.items():
        cv2.imwrite(os.path.join(dir_debug, nome), img)
