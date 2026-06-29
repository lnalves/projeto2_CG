"""Medição semiautomática por caminho de custo mínimo (live-wire) — ADR 0001
A classe `LiveWireTool` é a interface HighGUI de cliques.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.graph import route_through_array

from display import ajustar_exibicao
from log import log
from measure import medir_caminho
from render import ResultadoPlantula


def _impar(n: int, minimo: int = 1) -> int:
    """Garante valor ímpar >= `minimo` (exigido por kernels)."""
    n = max(minimo, int(n))
    return n if n % 2 == 1 else n + 1


def _blackhat_raw(luminancia: np.ndarray, kernel_size: int) -> np.ndarray:
    """Aplica black-hat morfológico e normaliza para [0, 1].

    Retorna float64: valores altos = estruturas finas escuras (filamento).
    """
    k = _impar(kernel_size, minimo=3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bh = cv2.morphologyEx(luminancia, cv2.MORPH_BLACKHAT, kernel).astype(np.float64)
    pico = bh.max()
    return bh / pico if pico > 0 else bh


def _mascara_blackhat(luminancia: np.ndarray, kernel_size: int, limiar: int) -> np.ndarray:
    """Máscara binária do filamento por black-hat seguido de threshold.

    O `_blackhat_raw` devolve valores normalizados em [0, 1]; aqui ele é
    reescalado para 0–255 para casar com `limiar_blackhat` (escala 0–255).
    `limiar <= 0` usa Otsu automático.
    """
    bh = _blackhat_raw(luminancia, kernel_size)
    bh8 = (bh * 255.0).astype(np.uint8)
    if limiar <= 0:
        _, mascara = cv2.threshold(bh8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, mascara = cv2.threshold(bh8, int(limiar), 1, cv2.THRESH_BINARY)
    return mascara.astype(np.uint8)


# --- Imagem de custo e caminho ---------------------------------------------

def imagem_custo(canais, config) -> np.ndarray:
    """Custo por pixel: baixo no filamento (black-hat alto), alto no papel.

    Usa o canal L do Lab (luminância perceptual equalizada) em vez do HSV V,
    pois o L* é perceptual e o CLAHE já realçou o contraste local.
    """
    lw = config.livewire
    realce = _blackhat_raw(canais.lab_l, lw.blackhat_kernel)
    custo = (1.0 - realce) ** float(lw.custo_gamma)
    return custo + 1e-3  # piso evita custo zero (degenera o Dijkstra)


def caminho_custo_minimo(
    custo: np.ndarray, origem: tuple, destino: tuple, margem: int, area_max: int = 2_000_000
) -> list | None:
    """Caminho de menor custo (lista de (y, x)) entre origem e destino.

    O roteamento roda só num recorte ao redor dos dois pontos (+ `margem`),
    para ser rápido e local. Retorna None se a área do recorte exceder
    `area_max` (OOM proteção).
    """
    h, w = custo.shape
    (y0, x0), (y1, x1) = origem, destino
    ymin = max(0, min(y0, y1) - margem)
    ymax = min(h, max(y0, y1) + margem + 1)
    xmin = max(0, min(x0, x1) - margem)
    xmax = min(w, max(x0, x1) + margem + 1)

    area = (ymax - ymin) * (xmax - xmin)
    if area > area_max:
        log.warning(
            "Recorte muito grande para roteamento: {}px² (limite {}px²). "
            "Aumente a margem ou reduza a imagem.",
            area, area_max,
        )
        return None

    sub = np.ascontiguousarray(custo[ymin:ymax, xmin:xmax])
    inicio = (y0 - ymin, x0 - xmin)
    fim = (y1 - ymin, x1 - xmin)
    caminho, _ = route_through_array(
        sub, inicio, fim, fully_connected=True, geometric=True,
    )
    return [(int(p[0] + ymin), int(p[1] + xmin)) for p in caminho]


# --- Colo ao longo do caminho ----------------------------------------------

def mapa_espessura(canais, config) -> np.ndarray:
    """Espessura local (distância ao fundo) da máscara de filamento black-hat."""
    lw = config.livewire
    mascara = _mascara_blackhat(canais.lab_l, lw.blackhat_kernel, int(lw.limiar_blackhat))
    return cv2.distanceTransform(mascara, cv2.DIST_L2, 5)


def indice_colo(caminho: list, espessura: np.ndarray) -> int:
    """Índice do colo no caminho: maior queda da espessura (mesma definição
    geométrica da Etapa 5), ignorando as extremidades.
    """
    n = len(caminho)
    if n < 5:
        return n // 2
    if n < 20:
        log.debug("Caminho curto ({}px) — colo pode ser impreciso.", n)
    esp = np.array([espessura[y, x] for (y, x) in caminho], dtype=np.float64)
    jan = max(3, n // 15)
    suave = np.convolve(esp, np.ones(jan) / jan, mode="same")
    ini, fim = int(0.15 * n), int(0.85 * n)
    if fim - ini < 2:
        return n // 2
    grad = np.diff(suave)
    return ini + int(np.argmin(grad[ini:fim])) + 1


# --- Sementes (snap do topo) -----------------------------------------------

def detectar_roi_papel(canais, config) -> np.ndarray | None:
    """Máscara (uint8 0/255) da região do papel-filtro, ou None se não confiável.

    O papel é a maior mancha clara (V alto) e pouco saturada (branca) da cena;
    régua, rótulos e bordas da caixa ficam de fora. Retorna o fecho convexo
    dessa mancha preenchido. Devolve None (sem restrição) se a ROI estiver
    desligada ou cobrir menos que `roi_area_min_frac` da imagem — assim a
    filtragem nunca descarta sementes por uma detecção de papel ruim.
    """
    lw = config.livewire
    if not lw.roi_papel:
        return None

    h, w = canais.valor.shape[:2]
    candidato = (
        (canais.valor >= int(lw.roi_valor_min)) & (canais.saturacao <= int(lw.roi_sat_max))
    ).astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(lw.roi_morfologia_px),) * 2)
    candidato = cv2.morphologyEx(candidato, cv2.MORPH_CLOSE, k)
    candidato = cv2.morphologyEx(candidato, cv2.MORPH_OPEN, k)

    num, lab, stats, _ = cv2.connectedComponentsWithStats(candidato, connectivity=8)
    if num <= 1:
        log.warning("ROI do papel não encontrada — sem restrição.")
        return None

    maior = 1 + int(np.argmax([stats[i, cv2.CC_STAT_AREA] for i in range(1, num)]))
    area = int(stats[maior, cv2.CC_STAT_AREA])
    if area < lw.roi_area_min_frac * h * w:
        log.warning(
            "ROI do papel pouco confiável ({:.1%} da imagem < {:.0%}) — sem restrição.",
            area / (h * w), lw.roi_area_min_frac,
        )
        return None

    mascara = (lab == maior).astype(np.uint8) * 255
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(contornos, key=cv2.contourArea))
    roi = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(roi, [hull], -1, 255, -1)
    log.info("ROI do papel: {:.1%} da imagem.", area / (h * w))
    return roi


def detectar_sementes(canais, config) -> list:
    """Lista de plântulas-semente como arrays (M, 2) de pixels (y, x).

    A semente é escura (V baixo). Para não confundir com régua, rótulos e
    bordas da caixa — que também são escuros — os componentes são filtrados por:
      - abertura morfológica, que apaga traços finos (linhas da régua/borda);
      - faixa de área [min, max], que descarta ruído e sombras grandes;
      - razão de aspecto máxima, que descarta estruturas alongadas;
      - ROI do papel-filtro, que descarta o que está fora do papel.
    """
    lw = config.livewire
    _, sem = cv2.threshold(canais.valor, int(lw.limiar_semente), 255, cv2.THRESH_BINARY_INV)

    abertura = int(lw.semente_abertura_px)
    if abertura > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (abertura, abertura))
        sem = cv2.morphologyEx(sem, cv2.MORPH_OPEN, kernel)

    roi = detectar_roi_papel(canais, config)

    num, rot, stats, centroides = cv2.connectedComponentsWithStats(sem, connectivity=8)
    amin, amax = int(lw.semente_area_min), int(lw.semente_area_max)
    asp_max = float(lw.semente_aspecto_max)
    sementes = []
    fora_roi = 0
    for lbl in range(1, num):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if not (amin <= area <= amax):
            continue
        bw = stats[lbl, cv2.CC_STAT_WIDTH]
        bh = stats[lbl, cv2.CC_STAT_HEIGHT]
        aspecto = max(bw, bh) / max(1, min(bw, bh))
        if aspecto > asp_max:
            continue
        if roi is not None:
            cx, cy = centroides[lbl]
            if roi[int(round(cy)), int(round(cx))] == 0:
                fora_roi += 1
                continue
        sementes.append(np.argwhere(rot == lbl))
    log.info(
        "Detectadas {} sementes (área {}–{}px, aspecto ≤ {:.1f}, {} fora da ROI)",
        len(sementes), amin, amax, asp_max, fora_roi,
    )
    return sementes


def snap_topo(clique: tuple, sementes: list, raio: int) -> tuple:
    """Se o clique cair perto de uma semente, devolve a borda INFERIOR dela
    (a junção semente↔hipocótilo = topo). Caso contrário, devolve o clique.
    """
    cy, cx = clique
    alvo, melhor = None, float(raio) ** 2
    for s in sementes:
        d = (s[:, 0] - cy) ** 2 + (s[:, 1] - cx) ** 2
        dmin = float(d.min())
        if dmin < melhor:
            melhor, alvo = dmin, s
    if alvo is None:
        return (int(cy), int(cx))
    ymax = int(alvo[:, 0].max())
    xs = alvo[alvo[:, 0] == ymax][:, 1]
    snapped = (ymax, int(round(xs.mean())))
    log.debug("Snap topo: clique {} → semente em {}", (cy, cx), snapped)
    return snapped


# --- Medição de um par topo→ponta ------------------------------------------

def medir_par(
    topo: tuple, ponta: tuple, custo: np.ndarray, espessura: np.ndarray,
    escala_mm_px: float, config,
):
    """Calcula caminho, colo e medida de uma plântula a partir de 2 pontos.

    Retorna (medicao, caminho, colo) — `medicao` é um measure.ResultadoMedicao.
    Pode retornar None se o roteamento falhar (recorte muito grande).
    """
    lw = config.livewire
    caminho = caminho_custo_minimo(
        custo, topo, ponta, int(lw.margem_bbox_px), int(lw.bbox_area_max_px),
    )
    if caminho is None:
        log.error("Roteamento falhou entre topo {} e ponta {}", topo, ponta)
        return None
    idx = indice_colo(caminho, espessura)
    medicao = medir_caminho(caminho, idx, escala_mm_px, config)
    log.info("Plântula medida: {:.1f}mm (colo no índice {} do caminho)", medicao.total_mm, idx)
    return medicao, caminho, caminho[idx]


# --- Ferramenta interativa (HighGUI) ----------------------------------------


class LiveWireTool:
    """Interface HighGUI interativa de cliques para medição live-wire.

    Controles:
      - Clique esquerdo 1º → topo (gruda na semente perto)
      - Clique esquerdo 2º → ponta → desenha caminho e mede
      - Botão direito  → reposiciona o colo da última plântula
      - 'z' → desfaz a última
      - ENTER / 'q' → finaliza, ESC → cancela tudo
    """

    def __init__(self, imagem, canais, escala_mm_px, config):
        self.imagem = imagem
        self.canais = canais
        self.escala_mm_px = escala_mm_px
        self.config = config
        self.ren = config.renderizacao

        self.custo = imagem_custo(canais, config)
        self.espessura = mapa_espessura(canais, config)
        self.sementes = detectar_sementes(canais, config)

        self.disp, self.escala_disp = ajustar_exibicao(
            imagem, config.calibracao.largura_max_exibicao, config.calibracao.margem_tela,
        )
        self.janela = "Live-wire: topo->ponta | dir=colo | z=desfaz ENTER=ok ESC=cancela"
        cv2.namedWindow(self.janela, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.janela, self._on_mouse)

        self.fase = "topo"
        self.topo_atual: tuple | None = None
        self.resultados: list[ResultadoPlantula] = []
        self.cancelar = False

    # --- conversão de coordenadas ------------------------------------------

    def _orig(self, mx: int, my: int) -> tuple[int, int]:
        """Coordenadas da imagem original a partir do clique na tela."""
        return (int(round(my / self.escala_disp)), int(round(mx / self.escala_disp)))

    def _ponto_disp(self, yx: tuple) -> tuple[int, int]:
        """Coordenadas da tela a partir de um ponto (y, x) original."""
        return (int(round(yx[1] * self.escala_disp)), int(round(yx[0] * self.escala_disp)))

    # --- callback de mouse -------------------------------------------------

    def _on_mouse(self, evento, mx, my, flags, param):
        if evento == cv2.EVENT_LBUTTONDOWN:
            p = self._orig(mx, my)
            if self.fase == "topo":
                self.topo_atual = snap_topo(p, self.sementes, self.config.livewire.snap_raio_px)
                self.fase = "ponta"
                log.debug("Topo marcado em {}", self.topo_atual)
            else:
                resultado = self._medir_par(self.topo_atual, p)
                if resultado is not None:
                    self.resultados.append(resultado)
                self.fase = "topo"
                self.topo_atual = None
        elif evento == cv2.EVENT_RBUTTONDOWN and self.resultados:
            p = self._orig(mx, my)
            self._reposicionar_colo(p)

    def _medir_par(self, topo, ponta) -> ResultadoPlantula | None:
        med = medir_par(topo, ponta, self.custo, self.espessura, self.escala_mm_px, self.config)
        if med is None:
            return None
        medicao, caminho, colo = med
        return ResultadoPlantula(
            id=len(self.resultados) + 1,
            topo=topo,
            colo=colo,
            ponta=ponta,
            medicao=medicao,
        )

    def _reposicionar_colo(self, clique: tuple):
        """Reposiciona o colo da última plântula no ponto do caminho mais
        próximo do clique, e remede."""
        r = self.resultados[-1]
        caminho = r.medicao.caminho1[:-1] + r.medicao.caminho2
        d = [(yy - clique[0]) ** 2 + (xx - clique[1]) ** 2 for (yy, xx) in caminho]
        idx = int(np.argmin(d))
        nova = medir_caminho(caminho, idx, self.escala_mm_px, self.config)
        self.resultados[-1] = ResultadoPlantula(
            id=r.id, topo=r.topo, colo=caminho[idx], ponta=r.ponta, medicao=nova,
        )
        log.info("Colo da plântula #{} reposicionado.", r.id)

    # --- renderização ------------------------------------------------------

    def _desenhar(self, tela):
        for r in self.resultados:
            for caminho, cor in (
                (r.medicao.caminho1, self.ren.cor_seg1_bgr),
                (r.medicao.caminho2, self.ren.cor_seg2_bgr),
            ):
                if len(caminho) > 1:
                    pts = np.array([self._ponto_disp(p) for p in caminho], np.int32).reshape(-1, 1, 2)
                    cv2.polylines(tela, [pts], False, cor, 2)
            cv2.circle(tela, self._ponto_disp(r.topo), 4, (0, 220, 0), -1)
            cv2.circle(tela, self._ponto_disp(r.colo), 4, (0, 0, 255), -1)
            cv2.circle(tela, self._ponto_disp(r.ponta), 4, (255, 0, 0), -1)
            cv2.putText(
                tela, f"#{r.id}: {r.medicao.total_mm:.1f}mm",
                (self._ponto_disp(r.topo)[0] + 6, self._ponto_disp(r.topo)[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1, cv2.LINE_AA,
            )
        if self.topo_atual is not None:
            cv2.circle(tela, self._ponto_disp(self.topo_atual), 5, (0, 255, 255), 2)
        cv2.putText(
            tela, f"plantulas: {len(self.resultados)} | fase: {self.fase}",
            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
        )

    # --- loop principal ----------------------------------------------------

    def run(self) -> list[ResultadoPlantula]:
        log.info("Live-wire: coleta de plântulas iniciada.")
        try:
            while True:
                tela = self.disp.copy()
                self._desenhar(tela)
                cv2.imshow(self.janela, tela)

                tecla = cv2.waitKey(20) & 0xFF
                if tecla == 27:
                    self.cancelar = True
                    log.info("Live-wire cancelado pelo usuário.")
                    break
                if tecla in (13, 10) or tecla == ord("q"):
                    log.info("Live-wire finalizado com {} plântulas.", len(self.resultados))
                    break
                if tecla == ord("z") and self.resultados:
                    self.resultados.pop()
                    for i, r in enumerate(self.resultados):
                        r.id = i + 1
                    self.fase = "topo"
                    self.topo_atual = None
                    log.debug("Última plântula desfeita.")
        finally:
            cv2.destroyWindow(self.janela)

        return [] if self.cancelar else self.resultados


# --- Função de compatibilidade (delega para a classe) ----------------------

def ferramenta_livewire(imagem, canais, escala_mm_px, config) -> list:
    """Compatibilidade: delega para LiveWireTool."""
    return LiveWireTool(imagem, canais, escala_mm_px, config).run()
