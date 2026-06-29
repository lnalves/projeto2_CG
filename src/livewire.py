"""Medição semiautomática por caminho de custo mínimo (live-wire) — ADR 0001.

Em vez de depender de uma segmentação binária limpa do filamento branco
(inviável sobre papel branco), o usuário clica **topo** e **ponta** de cada
plântula e o comprimento é o **caminho de custo mínimo** entre os dois cliques
sobre uma imagem de custo derivada do realce black-hat: o filamento fraco fica
"barato" e o papel "caro", então o caminho segue a curva real do filamento mesmo
onde o threshold falharia. O **colo** é detectado automaticamente ao longo desse
caminho (pela espessura) e a semente serve de **snap** para o clique do topo.

Funções-núcleo (testáveis sem GUI): `imagem_custo`, `caminho_custo_minimo`,
`indice_colo`, `detectar_sementes`/`snap_topo`, `medir_par`.
A `ferramenta_livewire` é a interface HighGUI de cliques.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.graph import route_through_array

from measure import medir_caminho
from render import ResultadoPlantula


def _impar(n: int, minimo: int = 1) -> int:
    """Garante valor ímpar >= `minimo` (exigido por kernels)."""
    n = max(minimo, int(n))
    return n if n % 2 == 1 else n + 1


def _blackhat_filamento(luminancia, lw) -> np.ndarray:
    """Máscara binária do filamento por black-hat (estruturas finas escuras)."""
    k = _impar(lw.blackhat_kernel, minimo=3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bh = cv2.morphologyEx(luminancia, cv2.MORPH_BLACKHAT, kernel)
    if int(lw.limiar_blackhat) <= 0:
        _, fila = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, fila = cv2.threshold(bh, int(lw.limiar_blackhat), 255, cv2.THRESH_BINARY)
    return fila


# --- Imagem de custo e caminho ---------------------------------------------

def imagem_custo(canais, config) -> np.ndarray:
    """Custo por pixel: baixo no filamento (black-hat alto), alto no papel."""
    lw = config.livewire
    k = int(lw.blackhat_kernel) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bh = cv2.morphologyEx(canais.valor, cv2.MORPH_BLACKHAT, kernel).astype(np.float64)
    pico = bh.max()
    realce = bh / pico if pico > 0 else bh
    # filamento (realce ~1) → custo ~0; papel (realce ~0) → custo ~1.
    custo = (1.0 - realce) ** float(lw.custo_gamma)
    return custo + 1e-3  # piso evita custo zero (degenera o Dijkstra)


def caminho_custo_minimo(custo, origem, destino, margem) -> list:
    """Caminho de menor custo (lista de (y, x)) entre origem e destino.

    O roteamento roda só num recorte ao redor dos dois pontos (+ `margem`),
    para ser rápido e local.
    """
    h, w = custo.shape
    (y0, x0), (y1, x1) = origem, destino
    ymin = max(0, min(y0, y1) - margem)
    ymax = min(h, max(y0, y1) + margem + 1)
    xmin = max(0, min(x0, x1) - margem)
    xmax = min(w, max(x0, x1) + margem + 1)
    sub = np.ascontiguousarray(custo[ymin:ymax, xmin:xmax])

    inicio = (y0 - ymin, x0 - xmin)
    fim = (y1 - ymin, x1 - xmin)
    caminho, _ = route_through_array(
        sub, inicio, fim, fully_connected=True, geometric=True
    )
    return [(int(p[0] + ymin), int(p[1] + xmin)) for p in caminho]


# --- Colo ao longo do caminho ----------------------------------------------

def mapa_espessura(canais, config) -> np.ndarray:
    """Espessura local (distância ao fundo) da máscara de filamento black-hat."""
    mascara = _blackhat_filamento(canais.valor, config.livewire)
    return cv2.distanceTransform(mascara, cv2.DIST_L2, 5)


def indice_colo(caminho, espessura) -> int:
    """Índice do colo no caminho: maior queda da espessura (mesma definição
    geométrica da Etapa 5), ignorando as extremidades.
    """
    n = len(caminho)
    if n < 5:
        return n // 2
    esp = np.array([espessura[y, x] for (y, x) in caminho], dtype=np.float64)
    jan = max(3, n // 15)
    suave = np.convolve(esp, np.ones(jan) / jan, mode="same")
    ini, fim = int(0.15 * n), int(0.85 * n)
    if fim - ini < 2:
        return n // 2
    grad = np.diff(suave)
    return ini + int(np.argmin(grad[ini:fim])) + 1


# --- Sementes (snap do topo) -----------------------------------------------

def detectar_sementes(canais, config) -> list:
    """Lista de plântulas-semente como arrays (M, 2) de pixels (y, x)."""
    lw = config.livewire
    _, sem = cv2.threshold(canais.valor, int(lw.limiar_semente), 255, cv2.THRESH_BINARY_INV)
    num, rot, stats, _ = cv2.connectedComponentsWithStats(sem, connectivity=8)
    sementes = []
    minimo = int(lw.semente_area_min)
    for lbl in range(1, num):
        if stats[lbl, cv2.CC_STAT_AREA] >= minimo:
            sementes.append(np.argwhere(rot == lbl))
    return sementes


def snap_topo(clique, sementes, raio) -> tuple:
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
    return (ymax, int(round(xs.mean())))


# --- Medição de um par topo→ponta ------------------------------------------

def medir_par(topo, ponta, custo, espessura, escala_mm_px, config):
    """Calcula caminho, colo e medida de uma plântula a partir de 2 pontos.

    Retorna (medicao, caminho, colo) — `medicao` é um measure.ResultadoMedicao.
    """
    caminho = caminho_custo_minimo(custo, topo, ponta, int(config.livewire.margem_bbox_px))
    idx = indice_colo(caminho, espessura)
    medicao = medir_caminho(caminho, idx, escala_mm_px, config)
    return medicao, caminho, caminho[idx]


# --- Ferramenta interativa (HighGUI) ---------------------------------------

def _ajustar_exibicao(imagem, largura_max):
    h, w = imagem.shape[:2]
    if w <= largura_max:
        return imagem, 1.0
    s = largura_max / float(w)
    return cv2.resize(imagem, (largura_max, int(round(h * s))), interpolation=cv2.INTER_AREA), s


def ferramenta_livewire(imagem, canais, escala_mm_px, config) -> list:
    """Coleta plântulas por cliques (topo→ponta) e devolve [ResultadoPlantula].

    Controles: clique 1 = topo (gruda na semente perto), clique 2 = ponta →
    desenha o caminho e mede. Botão direito reposiciona o colo da última
    plântula. 'z' desfaz a última, ENTER/'q' finaliza, ESC cancela tudo.
    """
    custo = imagem_custo(canais, config)
    espessura = mapa_espessura(canais, config)
    sementes = detectar_sementes(canais, config)
    ren = config.renderizacao

    disp, escala_disp = _ajustar_exibicao(imagem, config.calibracao.largura_max_exibicao)
    janela = "Live-wire: topo->ponta | dir=colo | z=desfaz ENTER=ok ESC=cancela"
    cv2.namedWindow(janela, cv2.WINDOW_AUTOSIZE)

    estado = {"fase": "topo", "topo": None, "cancelar": False}
    resultados: list = []

    def orig(mx, my):
        return (int(round(my / escala_disp)), int(round(mx / escala_disp)))

    def ao_clicar(evento, mx, my, flags, param):
        if evento == cv2.EVENT_LBUTTONDOWN:
            p = orig(mx, my)
            if estado["fase"] == "topo":
                estado["topo"] = snap_topo(p, sementes, config.livewire.snap_raio_px)
                estado["fase"] = "ponta"
            else:
                med, caminho, colo = medir_par(
                    estado["topo"], p, custo, espessura, escala_mm_px, config
                )
                resultados.append(ResultadoPlantula(
                    id=len(resultados) + 1, topo=estado["topo"], colo=colo,
                    ponta=p, medicao=med,
                ))
                estado["fase"] = "topo"
                estado["topo"] = None
        elif evento == cv2.EVENT_RBUTTONDOWN and resultados:
            # Reposiciona o colo da última plântula no ponto do caminho mais
            # próximo do clique, e remede.
            p = orig(mx, my)
            r = resultados[-1]
            caminho = r.medicao.caminho1[:-1] + r.medicao.caminho2
            d = [(yy - p[0]) ** 2 + (xx - p[1]) ** 2 for (yy, xx) in caminho]
            idx = int(np.argmin(d))
            nova = medir_caminho(caminho, idx, escala_mm_px, config)
            resultados[-1] = ResultadoPlantula(
                id=r.id, topo=r.topo, colo=caminho[idx], ponta=r.ponta, medicao=nova,
            )

    cv2.setMouseCallback(janela, ao_clicar)

    def ponto_disp(yx):
        return (int(round(yx[1] * escala_disp)), int(round(yx[0] * escala_disp)))

    try:
        while True:
            tela = disp.copy()
            for r in resultados:
                for caminho, cor in (
                    (r.medicao.caminho1, ren.cor_seg1_bgr),
                    (r.medicao.caminho2, ren.cor_seg2_bgr),
                ):
                    if len(caminho) > 1:
                        pts = np.array([ponto_disp(p) for p in caminho], np.int32).reshape(-1, 1, 2)
                        cv2.polylines(tela, [pts], False, cor, 2)
                cv2.circle(tela, ponto_disp(r.topo), 4, (0, 220, 0), -1)
                cv2.circle(tela, ponto_disp(r.colo), 4, (0, 0, 255), -1)
                cv2.circle(tela, ponto_disp(r.ponta), 4, (255, 0, 0), -1)
                cv2.putText(tela, f"#{r.id}: {r.medicao.total_mm:.1f}mm",
                            (ponto_disp(r.topo)[0] + 6, ponto_disp(r.topo)[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1, cv2.LINE_AA)
            if estado["topo"] is not None:
                cv2.circle(tela, ponto_disp(estado["topo"]), 5, (0, 255, 255), 2)
            cv2.putText(tela, f"plantulas: {len(resultados)} | fase: {estado['fase']}",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow(janela, tela)

            tecla = cv2.waitKey(20) & 0xFF
            if tecla == 27:  # ESC
                estado["cancelar"] = True
                break
            if tecla in (13, 10) or tecla == ord("q"):  # ENTER / q
                break
            if tecla == ord("z") and resultados:  # desfaz
                resultados.pop()
                for i, r in enumerate(resultados):
                    r.id = i + 1
                estado["fase"] = "topo"
                estado["topo"] = None
    finally:
        cv2.destroyWindow(janela)

    return [] if estado["cancelar"] else resultados
