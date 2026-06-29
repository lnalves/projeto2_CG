"""Utilitários de exibição: ajustar imagens para caber na tela do usuário.

As janelas interativas (calibração e live-wire) usam `WINDOW_AUTOSIZE`, então o
tamanho da janela é o tamanho da imagem exibida. Como as fotos são em retrato
(mais altas que largas), limitar só a largura deixava a janela mais alta que o
monitor. Aqui a imagem é reduzida para caber na **largura e na altura** úteis da
tela, com uma margem para barra de título e barra de tarefas.
"""

from __future__ import annotations

import cv2

from log import log

_TELA_PADRAO = (1280, 720)  # fallback se a tela não puder ser detectada


def tamanho_tela(padrao: tuple[int, int] = _TELA_PADRAO) -> tuple[int, int]:
    """(largura, altura) da tela em px. Usa tkinter; cai no padrão se indisponível."""
    try:
        import tkinter

        root = tkinter.Tk()
        root.withdraw()
        try:
            return root.winfo_screenwidth(), root.winfo_screenheight()
        finally:
            root.destroy()
    except Exception as exc:  # tkinter ausente, sem display, etc.
        log.debug("Tamanho da tela indisponível ({}) — usando padrão {}.", exc, padrao)
        return padrao


def ajustar_exibicao(imagem, largura_max: int = 0, margem_tela: float = 0.9):
    """Reduz a imagem para caber na tela (largura e altura), preservando o aspecto.

    Args:
        imagem: imagem BGR/cinza a exibir.
        largura_max: teto adicional de largura (px); 0 = sem teto além da tela.
        margem_tela: fração da tela usada (deixa espaço p/ barra de título/tarefas).

    Returns:
        (imagem_exibicao, escala). `escala` (<=1) multiplica coordenadas da
        imagem original para chegar nas da imagem exibida.
    """
    h, w = imagem.shape[:2]
    tela_w, tela_h = tamanho_tela()
    lim_w = int(tela_w * margem_tela)
    lim_h = int(tela_h * margem_tela)
    if largura_max and largura_max > 0:
        lim_w = min(lim_w, int(largura_max))

    escala = min(1.0, lim_w / float(w), lim_h / float(h))
    if escala >= 1.0:
        return imagem, 1.0

    nova = cv2.resize(
        imagem, (int(round(w * escala)), int(round(h * escala))),
        interpolation=cv2.INTER_AREA,
    )
    log.debug("Exibição: {}x{} → {}x{} (escala={:.3f}, tela {}x{})",
              w, h, nova.shape[1], nova.shape[0], escala, tela_w, tela_h)
    return nova, escala
