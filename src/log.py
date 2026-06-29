"""Configuração centralizada de logging (loguru).

Uso:
    from log import log

    log.info("Calibração concluída: {:.4f} mm/px", escala)
    log.debug("Forma da imagem: {}", img.shape)
"""

from __future__ import annotations

import sys

from loguru import logger

# Remove o handler padrão e adiciona um formatado.
logger.remove()
logger.add(
    sys.stderr,
    format=(
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level.icon}</level> "
        "<level>{message}</level>"
    ),
    level="INFO",
    colorize=True,
    backtrace=True,
    diagnose=False,
)

log = logger
