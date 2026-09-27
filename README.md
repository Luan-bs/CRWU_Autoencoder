# CWRU Autoencoder

Autoencoder convolucional 1D para deteccao nao supervisionada de falhas em dados
de rolamentos CWRU, acompanhado de uma analise exploratoria executavel como
notebook no VS Code.

## Dados disponiveis

- `data/CWRU_48k_load_1_CNN_data.npz`: arrays `data` com formato `(4600, 32, 32)`
	e `labels` com 10 classes equilibradas (460 amostras por classe). O treino
	transforma cada matriz em uma sequencia de 1024 valores para o Conv1D.
- `data/feature_time_48k_2048_load_1.csv`: 2.300 linhas de features agregadas,
	com 230 amostras por classe. Este CSV e o NPZ tem quantidades diferentes e
	devem ser analisados separadamente.
- `data/raw/`: arquivos MATLAB originais do conjunto CWRU.

## Instalacao e execucao

Com [uv](https://docs.astral.sh/uv/):

```powershell
uv sync
uv run python src/analise_dados.py
uv run python main.py --epochs 30
```

`src/analise_dados.py` esta dividido em celulas `# %%`; abra o arquivo no VS Code
e execute as celulas individualmente para explorar distribuicoes, features,
matrizes do NPZ e sinais brutos `.mat`.

O `main.py` e autocontido: treina somente com amostras normais, reserva normais
para calibrar o limiar `media + 3 * desvio padrao` e calcula ROC-AUC, matriz de
confusao, precision, recall, F1, MSE e MAE em amostras de teste reservadas.
Os pesos, relatorio e graficos sao gravados em `outputs/`. Opcoes como `--data`,
`--batch-size`, `--learning-rate` e `--output-dir` podem ser consultadas com
`uv run python main.py --help`.

