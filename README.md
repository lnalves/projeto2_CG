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
python src/main.py --image data/exemplo.jpg --out output
```

Opções principais:

| Flag | Descrição |
|---|---|
| `--image, -i` | Caminho da foto de entrada. |
| `--out, -o` | Pasta de saída (padrão: `output`). |
| `--known-mm` | Distância real (mm) entre os 2 pontos da calibração manual. |
| `--ruler-roi` | ROI da régua para detecção automática (`x,y,w,h`). |
| `--debug` | Salva imagens intermediárias em `<out>/debug/`. |

Veja todas as opções com `python src/main.py --help`.

## Estrutura

```
data/                imagens de entrada
output/              imagens anotadas + tabelas geradas
src/
├── main.py          orquestra o pipeline
├── config.py        parâmetros ajustáveis
├── calibration.py   régua → escala (px/mm)
├── preprocess.py    pré-processamento e realce
├── segmentation.py  máscara + separação de plântulas
├── skeleton.py      esqueleto + transformada de distância + 3 pontos
├── measure.py       comprimento ao longo do caminho real
└── render.py        marcações na imagem + tabela
```
