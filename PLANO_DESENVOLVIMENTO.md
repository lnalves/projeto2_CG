# Plano de Desenvolvimento — Medição de Plântulas de Alface

**Disciplina:** Computação Gráfica (UNISC) — Prof. Rafael Peiter
**Entrega e apresentação:** 29/06/2026
**Grupo:** até 6 pessoas

---

## 1. Visão geral do trabalho

O objetivo é, a partir de uma foto de plântulas de alface sobre papel-filtro (com uma régua na cena), medir **digitalmente** o comprimento de cada plântula, dividido em **dois segmentos**, seguindo o **caminho real** (curvo) da estrutura — não a linha reta entre pontos.

Para cada plântula:

- **Segmento 1 (hipocótilo):** do **topo da estrutura branca** até o **ponto de estrangulamento**, acompanhando o filamento.
- **Segmento 2 (raiz):** do **ponto de estrangulamento** até a **extremidade da raiz**, mesmo curvada ou enrolada.
- **Total:** soma dos dois segmentos.

### Terminologia botânica (importante para a apresentação)

A "estrutura branca" é o **hipocótilo** (caule jovem, esbranquiçado). No topo fica a semente/cotilédones (parte escura e amarelada). O **ponto de estrangulamento** é o **colo** da plântula — a transição hipocótilo → raiz, onde o filamento afina e muda de textura. Abaixo está a **raiz** (radícula), tipicamente mais fina e frequentemente curvada. Saber nomear isso ajuda na arguição individual do professor.

### Requisitos obrigatórios da especificação

- Identificar corretamente: topo da estrutura branca, ponto de estrangulamento e ponta da raiz.
- Dividir cada plântula em 2 segmentos e medir o **caminho real** (curvas, inclinações, partes enroladas).
- **Calibrar pela régua** da imagem (ou outro método) → resultado em **mm/cm**.
- Gerar **imagem com marcações visíveis** dos pontos medidos.
- Apresentar os resultados em **tabela** (id da plântula + medidas).
- **OpenCV é obrigatório**; linguagem livre.

---

## 2. Escolha da linguagem

### Recomendação: **Python**

É a melhor escolha para este trabalho, e por boa margem:

- **OpenCV em Python** (`opencv-python`) é maduro, com a mesma API do C++ porém muito mais rápido de prototipar — essencial num projeto com várias etapas de tentativa-e-erro de visão computacional.
- O ecossistema científico (NumPy, SciPy, scikit-image, pandas, matplotlib) cobre exatamente o que falta no OpenCV puro: **esqueletização robusta**, **transformada de distância**, **menor caminho em grafo** e **geração de tabelas/CSV**.
- Curva de aprendizado baixa, fácil de dividir entre 6 pessoas, e código curto de explicar na apresentação.

### Alternativa: **C++**

OpenCV é nativo em C++ e tem desempenho superior. Vale considerar **apenas se** o grupo já domina C++ e quer performance. As desvantagens para este caso: build/CMake mais trabalhoso, esqueletização e grafos exigem mais código manual (ou `opencv_contrib`), e a iteração é mais lenta. Para o escopo (uma imagem por vez, sem tempo real), o ganho de performance é irrelevante.

> **Decisão sugerida:** Python 3.10+. O restante do plano assume Python.

---

## 3. Stack de bibliotecas e frameworks

| Biblioteca | Papel no projeto | Por quê |
|---|---|---|
| **opencv-python** (+ **opencv-contrib-python**) | Obrigatória. Leitura/escrita de imagem, conversão de cor, threshold, morfologia, contornos, desenho das marcações. `ximgproc.thinning` (no contrib) p/ esqueleto. | Requisito do trabalho; base de todo o pipeline. |
| **NumPy** | Manipulação dos arrays de pixels e máscaras. | Base de tudo no Python científico. |
| **scikit-image** | `skeletonize` / `medial_axis` (esqueleto 1px + raio local), análise de regiões (`regionprops`). | Esqueletização mais robusta e fácil que a do OpenCV puro. |
| **SciPy** | `ndimage` (rotulagem, distância), `sparse.csgraph` (menor caminho no esqueleto). | Mede o comprimento ao longo do caminho real via grafo. |
| **pandas** | Montar a tabela de resultados e exportar CSV/Excel. | Atende o critério "organização dos dados em tabela". |
| **matplotlib** | Visualizações/figuras para o relatório e conferência. | Útil para debug e slides. |
| **networkx** (opcional) | Alternativa amigável ao SciPy para grafos do esqueleto. | Opcional; escolha entre ele ou `csgraph`. |

### Interface (opcional, mas recomendado para a calibração e correção manual)

- **OpenCV HighGUI** (`cv2.setMouseCallback`) — mais simples; o usuário clica nos dois pontos da régua para calibrar e, se preciso, ajusta pontos da plântula. Suficiente e fácil de demonstrar.
- **Tkinter** — GUI nativa do Python, sem dependência extra, se quiserem uma janela mais "de programa".
- **Streamlit** — se quiserem um app web bonito para a apresentação (upload da imagem → resultados). Mais vistoso, custo um pouco maior.

> Sugestão: comece com cliques via HighGUI para a calibração; só evolua para Tkinter/Streamlit se sobrar tempo.

### Instalação

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install opencv-python opencv-contrib-python numpy scipy scikit-image pandas matplotlib networkx
```

---

## 4. Estrutura sugerida do projeto

```
projeto2_CG/
├── data/                  # imagens de exemplo do ambiente virtual
├── output/                # imagens anotadas + tabelas geradas
├── src/
│   ├── main.py            # orquestra o pipeline
│   ├── calibration.py     # régua → escala (px/mm)
│   ├── preprocess.py      # pré-processamento e realce
│   ├── segmentation.py    # máscara + separação de plântulas
│   ├── skeleton.py        # esqueleto + grafo + pontos
│   ├── measure.py         # comprimento ao longo do caminho
│   └── render.py          # marcações na imagem + tabela
├── requirements.txt
└── README.md
```

---

## 5. Pipeline técnico (etapa por etapa)

O coração do trabalho. Cada etapa abaixo é também uma boa unidade de divisão entre os integrantes.

### 5.1 Aquisição e **calibração pela régua** (escala px → mm)

A medida final precisa estar em mm/cm, então a calibração vem primeiro.

- **Método recomendado (semiautomático, confiável):** mostrar a imagem e pedir que o usuário **clique em dois pontos da régua** de distância conhecida (ex.: marca de 1 cm a 10 cm = 90 mm). Calcula-se `escala_mm_por_px = distancia_mm / distancia_em_pixels`. Robusto a iluminação e ângulo.
- **Método automático (extra):** recortar a faixa da régua, aplicar threshold e detectar os "ticks" (picos periódicos na projeção horizontal/vertical) para estimar pixels por mm. Mais elegante, porém sensível a ruído — deixe como bônus.
- Cuidado com **perspectiva**: a régua está no topo e as plântulas mais embaixo, em outro plano. Se houver distorção perceptível, considere uma correção de perspectiva (`cv2.getPerspectiveTransform` / `warpPerspective`) usando os cantos da placa antes de medir. Para a maioria das fotos planas, a calibração simples basta.

### 5.2 Pré-processamento

O desafio é que a plântula é **esbranquiçada sobre papel branco**. Passos típicos:

- Converter para escala de cinza e/ou para espaços de cor úteis (**HSV** ou **Lab**) — o canal de saturação/luminância ajuda a separar a plântula translúcida e a semente escura do fundo.
- **CLAHE** (`cv2.createCLAHE`) para realçar contraste local.
- Suavização leve (`GaussianBlur`) para reduzir ruído antes de segmentar.

### 5.3 Segmentação das plântulas

- **Threshold adaptativo** (`adaptiveThreshold`) ou **Otsu** sobre o canal que melhor separa plântula × fundo. Como o fundo é claro e quase uniforme, often funciona bem combinar: detectar a **semente escura** (threshold baixo) e o **filamento** (realce + threshold) e unir.
- **Morfologia** (`morphologyEx` com `OPEN`/`CLOSE`) para remover ruído e fechar pequenas quebras no filamento.
- Resultado: **máscara binária** com as plântulas em branco.

> Dica: este é o ponto mais sensível. Vale ter parâmetros ajustáveis (trackbars do OpenCV) para calibrar a segmentação por imagem.

### 5.4 Separação das plântulas individuais

- **Componentes conexos** (`cv2.connectedComponentsWithStats`) para rotular cada plântula.
- Filtrar por **área mínima** (remove sujeira/ruído).
- Tratar casos de **plântulas que se cruzam/encostam**: se duas se fundem num só componente, pode ser necessário separação manual (clicar para indicar quais pixels pertencem a cada uma) ou heurística por proximidade da semente. Documentar como limitação conhecida.

### 5.5 Esqueletização (linha central do filamento)

- Aplicar **`skeletonize`** (scikit-image) ou **`cv2.ximgproc.thinning`** (contrib) na máscara de cada plântula → esqueleto de **1 pixel de largura** seguindo o eixo do filamento.
- Em paralelo, calcular a **transformada de distância** (`cv2.distanceTransform` ou `medial_axis(return_distance=True)`) — dá a **espessura local** ao longo do esqueleto, usada para achar o estrangulamento.
- Limpar **galhos espúrios** (pruning) do esqueleto: remover ramos curtos para ficar com o caminho principal.

### 5.6 Identificação dos três pontos

- **Topo da estrutura branca:** a extremidade do esqueleto mais próxima da **semente** (região escura/amarela detectada na segmentação) — ou simplesmente a ponta superior do hipocótilo.
- **Ponta da raiz:** a outra extremidade do esqueleto (endpoint mais distante do topo ao longo do caminho).
- **Ponto de estrangulamento (colo):** procurar ao longo do esqueleto onde a **espessura local** (da transformada de distância) tem uma **queda acentuada / mínimo local** — o hipocótilo é mais grosso, a raiz mais fina; a transição é o colo. Pode-se também combinar com mudança de cor/intensidade. Disponibilizar **ajuste manual** (clique) para corrigir quando a detecção automática errar.
- **Endpoints e bifurcações** do esqueleto são achados contando vizinhos de cada pixel (1 vizinho = ponta; ≥3 = bifurcação).

### 5.7 Medição ao longo do **caminho real**

- Modelar o esqueleto como **grafo**: cada pixel do esqueleto é um nó; ligações para vizinhos com peso **1 (ortogonal)** ou **√2 (diagonal)**.
- Calcular o **menor caminho** (Dijkstra via `scipy.sparse.csgraph` ou `networkx`) entre os pontos:
  - Segmento 1 = caminho topo → estrangulamento.
  - Segmento 2 = caminho estrangulamento → ponta da raiz.
- O **comprimento em pixels** é a soma dos pesos ao longo do caminho — isso captura curvas e partes enroladas, atendendo o critério "caminho real, não linha reta".
- Opcional: suavizar o caminho (spline) antes de somar, para reduzir o ruído de serrilhado do esqueleto e evitar superestimar o comprimento.

### 5.8 Conversão para unidades reais

- `comprimento_mm = comprimento_px × escala_mm_por_px` (da etapa 5.1).
- Apresentar em **mm** (ou cm) com 1–2 casas decimais.

### 5.9 Saída: imagem anotada + tabela

- **Imagem anotada** (`render.py`): desenhar o **caminho do segmento 1** numa cor e o **segmento 2** em outra (`cv2.polylines`), marcar os 3 pontos (`cv2.circle`) e rotular cada plântula com um id e o valor medido (`cv2.putText`). Salvar em `output/`. — atende "clareza visual / marcações visíveis".
- **Tabela** (`pandas` → CSV/Excel) com colunas: `id`, `segmento1_mm`, `segmento2_mm`, `total_mm`. — atende "organização dos dados".

---

## 6. Como cada critério de avaliação é atendido

| Critério da especificação | Onde é resolvido no plano |
|---|---|
| Identificação correta das estruturas | Etapa 5.6 (topo, estrangulamento, ponta) |
| Divisão adequada em 2 segmentos | Etapas 5.6 + 5.7 |
| Precisão (segue curvas/enrolados) | Etapa 5.7 (caminho via grafo no esqueleto) |
| Uso correto da escala (mm/cm) | Etapas 5.1 + 5.8 (calibração pela régua) |
| Organização dos dados em tabela | Etapa 5.9 (pandas → CSV) |
| Clareza visual (marcações) | Etapa 5.9 (imagem anotada) |
| Apresentação (todos falam) | Seção 7 (divisão por módulo) |

---

## 7. Cronograma até 29/06/2026 e divisão no grupo

Hoje é **04/06/2026** → ~3,5 semanas. Sugestão de fases:

**Semana 1 (04–10/06) — Fundação**

- Montar repositório, ambiente virtual e `requirements.txt`.
- Carregar e inspecionar as imagens de exemplo do ambiente virtual.
- Implementar a **calibração pela régua** (5.1) e o **carregamento** (5.2 inicial).
- **Enviar o nome dos integrantes ao professor (prazo: próxima semana).**

**Semana 2 (11–17/06) — Visão computacional central**

- Pré-processamento + segmentação (5.2–5.4).
- Esqueletização + transformada de distância (5.5).
- Primeira versão da detecção dos 3 pontos (5.6).

**Semana 3 (18–24/06) — Medição e saída**

- Medição por grafo / caminho real (5.7) + conversão de escala (5.8).
- Imagem anotada + tabela CSV (5.9).
- Ajuste fino dos parâmetros em todas as imagens de exemplo; validação dos resultados.

**Semana 4 (25–29/06) — Acabamento e apresentação**

- Tratamento de casos difíceis (plântulas cruzadas, detecção do colo).
- README, limpeza de código, comentários.
- **Slides + ensaio**: como todos precisam falar e responder perguntas, cada um apresenta o módulo que implementou.

### Divisão sugerida por integrante (até 6)

1. Calibração + perspectiva (5.1)
2. Pré-processamento + segmentação (5.2–5.3)
3. Separação de plântulas + esqueleto (5.4–5.5)
4. Detecção dos pontos / estrangulamento (5.6)
5. Medição por grafo + escala (5.7–5.8)
6. Render, tabela, GUI e integração `main.py` (5.9)

> Mesmo com a divisão, **todos devem entender o pipeline inteiro** — o professor faz perguntas específicas e desconta nota por respostas inadequadas.

---

## 8. Riscos e dicas

- **Plântula branca em fundo branco** é o maior risco da segmentação. Tenha trackbars para ajustar thresholds e teste em todas as imagens, não só numa.
- **Detecção automática do colo pode falhar.** Sempre ofereça **correção manual por clique** como rede de segurança — garante nota nos critérios de identificação e precisão.
- **Esqueleto serrilhado superestima o comprimento.** Suavizar o caminho (spline) deixa a medida mais realista.
- **Perspectiva/escala**: se a régua e as plântulas estão em planos diferentes, a conversão pode ter pequeno erro sistemático — mencione isso na apresentação como limitação consciente (mostra domínio do tema).
- **Validação**: meça manualmente 1–2 plântulas com régua na própria foto e compare com a saída do programa para reportar o erro aproximado.

---

## 9. Checklist de entrega

- [ ] Código-fonte funcionando com OpenCV, rodando sobre as imagens de exemplo.
- [ ] Calibração pela régua → resultados em mm/cm.
- [ ] Cada plântula com segmento 1, segmento 2 e total.
- [ ] Imagem(ns) anotada(s) com os pontos e caminhos marcados.
- [ ] Tabela (CSV/Excel) com id e medidas.
- [ ] README com instruções de execução.
- [ ] Nomes dos integrantes enviados ao professor (na primeira semana).
- [ ] Slides + ensaio com todos os integrantes falando.
