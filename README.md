# Medição de Plântulas de Alface

Trabalho de **Computação Gráfica** (UNISC — Prof. Rafael Peiter). A partir de
uma foto de plântulas de alface sobre papel-filtro,
o programa mede digitalmente o comprimento de cada plântula em **dois
segmentos**, seguindo o **caminho real curvo** da estrutura:

- **Segmento 1 (hipocótilo):** topo da estrutura branca → ponto de estrangulamento (colo).
- **Segmento 2 (raiz):** colo → ponta da raiz.
- **Total:** soma dos dois.

A medida sai em **mm** (calibrada pela régua) e o programa gera uma **imagem
anotada** + uma **tabela CSV** com id e medidas.

Como o filamento é branco translúcido sobre papel branco (contraste baixíssimo), a
medição é **semiautomática**: para cada plântula você clica **topo** e **ponta**, e
o programa traça o **caminho de custo mínimo** entre eles seguindo a curva real do
filamento, detectando o colo automaticamente. Veja
[CONTEXT.md](CONTEXT.md) e [docs/adr/0001](docs/adr/0001-medicao-semiautomatica-live-wire.md).

## Requisitos

- Python 3.10+

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

```bash
python src/main.py --image data/IMG_3196.png --out output
```

Fotos `.HEIC` são lidas se o `pillow-heif` estiver instalado; senão, converta para
PNG/JPG. Imagens grandes são reduzidas automaticamente (`--max-dim`).

### Medição por cliques

Abre uma janela com a foto. Para cada plântula:

| Ação | Efeito |
|---|---|
| Clique esquerdo (1º) | marca o **topo** (gruda na semente próxima) |
| Clique esquerdo (2º) | marca a **ponta** → traça o caminho e mede |
| Clique direito | reposiciona o **colo** da última plântula |
| `z` | desfaz a última plântula |
| `ENTER` / `q` | finaliza e salva |
| `ESC` | cancela |

A **calibração da régua** é feita **manualmente**: você clica em **2 pontos** de distância conhecida (ex.: marca de 1 cm a 10 cm na régua = 90 mm). Ajuste com `--known-mm` se não for 90 mm.

### Opções principais

| Flag | Descrição |
|---|---|
| `--image, -i` | Caminho da foto de entrada. |
| `--out, -o` | Pasta de saída (padrão: `output`). |
| `--known-mm` | Distância real (mm) entre os 2 pontos da calibração manual (padrão: 90). |
| `--max-dim` | Reduz a imagem p/ o maior lado ter no máx. N px (0 desliga). |
| `--debug` | Salva imagens intermediárias em `<out>/debug/`. |

Veja todas as opções com `python src/main.py --help`.

## Estrutura

```
data/                imagens de entrada
output/              imagens anotadas + tabelas geradas
src/
├── main.py          orquestra o pipeline
├── config.py        parâmetros ajustáveis
├── calibration.py   régua → escala (px/mm) + seleção da ROI
├── preprocess.py    pré-processamento e realce
├── livewire.py      medição por cliques (caminho de custo mínimo)
├── measure.py       comprimento ao longo do caminho
└── render.py        marcações na imagem + tabela
```
