"""Orquestra a medição semiautomática de plântulas de alface (live-wire).

Pipeline: carregar → calibrar (2 cliques com distância conhecida) → pré-processar
→ medir por cliques (topo→ponta, caminho de custo mínimo) → anotar/exportar.

Uso:
    python src/main.py --image data/IMG_3196.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import cv2
import numpy as np

from calibration import calibrar_detalhado
from config import carregar_config, config_padrao
from livewire import ferramenta_livewire
from log import log
from preprocess import preprocessar
from render import anotar_imagem, exportar_tabela


def carregar_imagem(caminho: str) -> np.ndarray:
    """Carrega a imagem como BGR. Suporta HEIC/HEIF se `pillow-heif` existir."""
    ext = Path(caminho).suffix.lower()
    if ext in (".heic", ".heif"):
        try:
            from PIL import Image
            from pillow_heif import register_heif_opener
        except ImportError:
            raise RuntimeError(
                f"'{caminho}' é HEIC e 'pillow-heif' não está instalado. "
                "Instale com `pip install pillow-heif` ou converta a foto para PNG/JPG."
            ) from None
        register_heif_opener()
        pil = Image.open(caminho).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    img = cv2.imread(caminho, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Não foi possível ler a imagem '{caminho}'.")
    log.info("Imagem carregada: {} ({})", caminho, img.shape[:2])
    return img


def redimensionar(imagem: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    """Reduz a imagem para que o maior lado tenha no máx. `max_dim`.

    Retorna (imagem, escala). A escala (<=1) multiplica coordenadas da imagem
    original para chegar nas da imagem reduzida.
    """
    if not max_dim or max_dim <= 0:
        return imagem, 1.0
    h, w = imagem.shape[:2]
    maior = max(h, w)
    if maior <= max_dim:
        return imagem, 1.0
    s = max_dim / float(maior)
    nova = cv2.resize(imagem, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    log.info("Imagem redimensionada: {} → {} (escala={:.3f})", (w, h), nova.shape[:2][::-1], s)
    return nova, s


def processar(imagem: np.ndarray, config, regua_roi=None, dir_debug=None):
    """Roda o pipeline e devolve (imagem_anotada, resultados).
    """
    cal = calibrar_detalhado(imagem, config, regua_roi=regua_roi)
    log.info("Escala da calibração: {:.5f} mm/px", cal.mm_por_px)

    canais = preprocessar(imagem, config, dir_debug=dir_debug)

    log.info("Iniciando live-wire. Clique topo→ponta para cada plântula.")
    resultados = ferramenta_livewire(imagem, canais, cal.mm_por_px, config)
    log.info("Medidas coletadas: {} plântulas.", len(resultados))

    anotada = anotar_imagem(imagem, resultados, config)
    return anotada, resultados


# --- CLI ---

@click.command()
@click.option("--image", "-i", required=True, help="Caminho da imagem de entrada.")
@click.option("--out", "-o", default="output", show_default=True, help="Pasta de saída.")
@click.option("--known-mm", type=float, default=90.0, show_default=True,
              help="Distância real (mm) entre os 2 pontos da calibração manual.")
@click.option("--max-dim", type=int, default=1600, show_default=True,
              help="Redimensiona a imagem p/ que o maior lado tenha no máx. N px (0 desliga).")
@click.option("--config", "-c", "config_path", default=None,
              help="Caminho do arquivo YAML de configuração.")
@click.option("--debug", is_flag=True, help="Salva imagens intermediárias em <out>/debug/.")
@click.version_option(version="0.1.0", prog_name="projeto2-cg")
def main(image, out, known_mm, max_dim, config_path, debug):
    """Mede plântulas de alface (hipocótilo + raiz) a partir de uma foto, em mm."""
    config = carregar_config(config_path) if config_path else config_padrao()
    config.debug = debug
    config.calibracao.distancia_conhecida_mm = known_mm

    if debug:
        log.remove()
        log.add(sys.stderr, level="DEBUG")

    log.info("Pipeline iniciado — distância calibração: {:.0f}mm", known_mm)

    try:
        imagem = carregar_imagem(image)
    except RuntimeError as exc:
        log.error("Erro ao carregar imagem: {}", exc)
        return 1

    imagem, escala = redimensionar(imagem, max_dim)

    out_path = Path(out)
    dir_debug = str(out_path / "debug") if debug else None

    try:
        anotada, resultados = processar(imagem, config, regua_roi=None, dir_debug=dir_debug)
    except RuntimeError as exc:
        log.error("Erro no pipeline: {}", exc)
        return 1

    out_path.mkdir(parents=True, exist_ok=True)
    nome_base = Path(image).stem
    caminho_img = out_path / f"{nome_base}_anotada.png"
    caminho_csv = out_path / f"{nome_base}_medidas.csv"

    cv2.imwrite(str(caminho_img), anotada)
    df = exportar_tabela(resultados, str(caminho_csv))

    click.echo("")
    if not df.empty:
        click.echo(df.to_string(index=False))
    else:
        click.echo("(nenhuma plântula medida)")
    click.echo(f"\nImagem anotada: {caminho_img}")
    click.echo(f"Tabela CSV:     {caminho_csv}")

    log.info("Pipeline concluído — saída em '{}'.", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
