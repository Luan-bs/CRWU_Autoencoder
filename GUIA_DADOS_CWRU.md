# Guia dos dados CWRU

Este guia explica, sem exigir conhecimento de mecânica, o que foi medido no
conjunto CWRU, o que significam os arquivos deste projeto e como eles se
relacionam com o autoencoder.

## A ideia em poucas palavras

Um motor gira apoiado por rolamentos. Quando um rolamento tem uma falha, a
vibração do conjunto pode mudar. A bancada CWRU mede essa vibração em condições
controladas e registra quais rolamentos estavam normais ou tinham falhas
artificiais. Isso permite comparar os sinais e testar métodos que tentam
detectar problemas antes de uma quebra.

Os dados não vêm de uma máquina industrial operando livremente durante anos.
São ensaios de laboratório, com falhas introduzidas de propósito e condições
conhecidas. Isso é ótimo para aprender e comparar métodos, mas não garante que
um modelo terá o mesmo desempenho em qualquer motor ou fábrica.

## Como o experimento funciona

O conjunto usa um motor com potência nominal de 2 HP, ligado a equipamentos que
controlam a carga e medem o funcionamento. Para os arquivos deste projeto, a
condição descrita pelo Kaggle é carga de 1 HP, rotação de 1.772 RPM e aquisição
de vibração a 48 kHz.

Potência nominal e carga são coisas diferentes: o motor é capaz de fornecer até
2 HP, mas este subconjunto foi medido com uma carga aplicada de 1 HP. Portanto,
os dados deste projeto não cobrem automaticamente todas as cargas e rotações do
experimento CWRU.

Os sensores chamados acelerômetros medem pequenas acelerações e vibrações. O
conjunto descreve três posições:

- **DE (Drive End):** lado do eixo de acionamento do motor.
- **FE (Fan End):** lado oposto, próximo à ventoinha do motor.
- **BA (Base):** estrutura/base onde o conjunto está apoiado.

Nos arquivos `.mat` presentes neste workspace encontramos variáveis de sinal
DE e FE. O CSV não possui uma coluna que identifique o sensor de cada linha;
por isso, não é possível descobrir o canal de uma linha do CSV apenas olhando
suas colunas.

### O que significa uma falha “semeada”

Os pesquisadores produziram pequenas marcas em um ponto do rolamento usando
EDM, uma técnica de usinagem por descarga elétrica. Não é uma falha que surgiu
naturalmente com o desgaste: ela foi criada de forma controlada para que seu
efeito pudesse ser estudado.

As marcas têm três diâmetros:

| Código | Diâmetro em polegadas | Aproximadamente em milímetros |
| --- | ---: | ---: |
| `007` | 0,007 | 0,178 |
| `014` | 0,014 | 0,356 |
| `021` | 0,021 | 0,533 |

Elas podem estar em três partes:

- **Ball / B:** elemento rolante, a “bolinha” que gira entre as pistas.
- **IR:** pista interna, ligada ao eixo.
- **OR:** pista externa, ligada à parte estacionária do rolamento.

## Frequência e janelas

**48 kHz** significa 48.000 medições por segundo. O computador transforma a
vibração contínua em uma lista de números, um valor por instante de medição.

Para calcular as features, o Kaggle divide o sinal em trechos de 2.048 pontos.
Cada trecho dura aproximadamente:

$$
\frac{2048}{48000} \approx 0{,}0427\ \text{segundos}
$$

Isso corresponde a cerca de 43 milissegundos; a descrição do Kaggle arredonda
para 0,04 s. Cada linha do CSV representa um trecho resumido, e não os 2.048
valores originais da vibração.

## Os arquivos deste projeto

### `data/raw/` — sinais brutos MATLAB

Os arquivos `.mat` guardam medições do experimento. Por exemplo:

- `B007_1_123.mat`: falha na esfera, diâmetro 0,007 polegada, carga 1 HP e
  registro CWRU 123.
- `IR014_1_175.mat`: falha na pista interna, diâmetro 0,014 polegada, carga
  1 HP e registro 175.
- `OR007_6_1_136.mat`: falha na pista externa, diâmetro 0,007 polegada,
  posição 6, carga 1 HP e registro 136.
- `Time_Normal_1_098.mat`: medição normal, carga 1 HP e registro 098.

Nos nomes de variáveis MATLAB, sufixos como `_DE_time` e `_FE_time` indicam
sinais de vibração dos canais Drive End e Fan End. Esses arquivos são a opção
mais próxima da forma de onda original para treinar um Conv1D sobre vibração.

### `data/feature_time_48k_2048_load_1.csv` — resumos numéricos

Este CSV tem 2.300 linhas: 230 para cada uma das 10 classes. Cada linha contém
nove valores numéricos e o rótulo `fault`:

| Coluna | O que resume |
| --- | --- |
| `max` / `min` | Maior e menor valor do trecho. |
| `mean` | Média dos valores; indica o nível médio e pode ser próxima de zero em vibrações centradas. |
| `sd` | Desvio padrão; indica o quanto os valores variam em torno da média. |
| `rms` | Amplitude eficaz, uma medida do nível geral de vibração. |
| `skewness` | Assimetria da distribuição dos valores. |
| `kurtosis` | Peso das caudas e presença de valores extremos ou impactos. |
| `crest` | Relação entre pico e RMS; picos fortes podem elevar esse valor. |
| `form` | Fator de forma, normalmente RMS dividido pela média do valor absoluto. |
| `fault` | Classe conhecida do trecho, usada como rótulo. |

As definições matemáticas exatas podem variar ligeiramente entre ferramentas.
Além disso, a unidade física das amplitudes deve ser confirmada na documentação
do sinal original; não se deve deduzi-la somente pelo nome das colunas.

As classes observadas são `Normal_1`, `Ball_007_1`, `Ball_014_1`,
`Ball_021_1`, as três classes `IR` correspondentes e as três classes `OR`.
Em `OR_007_6_1`, o `6` indica a posição da falha na pista externa; o último
`1` representa a condição de carga usada neste subconjunto.

O CSV é útil para estudar features ou treinar um modelo sobre essas features.
Ele não contém a sequência original de 2.048 pontos nem uma coluna de tempo,
arquivo de origem ou sensor. Portanto, não é possível reconstruir a vibração
bruta a partir dele.

### `data/CWRU_48k_load_1_CNN_data.npz` — dados pré-processados

Este arquivo compactado contém dois arrays: `data`, com formato
`(4600, 32, 32)`, e `labels`, com 4.600 rótulos. Há 460 exemplos em cada uma das
10 classes. Os rótulos guardam nomes como `Normal`, `Ball_007`, `IR_014` e
`OR_021`.

O nome sugere dados preparados para uma CNN, mas este repositório não documenta
como os valores foram transformados em matrizes `32×32`, nem como esses 4.600
exemplos se relacionam com as 2.300 linhas do CSV. Por isso, não se deve chamar
essas matrizes de forma de onda, imagem ou espectrograma sem localizar o código
que as gerou.

## Como o `main.py` usa os dados

O `main.py` atual lê somente o arquivo NPZ. Ele achata cada matriz `32×32` em
1.024 valores e os apresenta ao Conv1D como uma sequência. Em seguida:

1. Divide os exemplos de cada classe em 70%, 15% e 15%.
2. Usa somente os exemplos normais da primeira parte para treinar a
   reconstrução.
3. Usa normais reservados da segunda parte para calcular o limiar de anomalia.
4. Mantém a parte final de todas as classes para o teste. Para esse teste, os
   rótulos são convertidos em duas categorias: normal e falha.
5. Calcula o erro entre cada entrada e sua reconstrução. Um erro maior que o
   limiar é contado como alarme de falha.

O limiar é calculado como média do erro normal de calibração mais três desvios
padrão. ROC-AUC mede se os erros tendem a ser maiores nas falhas mesmo sem
escolher um limiar específico. Precision, recall, F1 e a matriz de confusão
avaliam os alarmes depois de aplicar o limiar.

## O que interpretar com cuidado

- O código atual não lê os `.mat` e não usa as nove features do CSV; usa apenas
  o NPZ.
- Achatar uma matriz `32×32` cria uma sequência de 1.024 posições, mas isso só
  tem significado temporal se a preparação do NPZ preservou essa relação.
- Uma divisão aleatória por amostra pode colocar trechos do mesmo ensaio tanto
  no treino quanto no teste. Se isso acontecer, o resultado pode parecer melhor
  do que a capacidade de reconhecer um ensaio ou rolamento nunca visto.
- Este conjunto é de uma condição específica de carga, rotação e frequência de
  aquisição. Um bom resultado aqui não prova generalização para outras
  condições de operação.
- Para um Conv1D diretamente sobre vibração, a abordagem mais interpretável é
  escolher um canal dos `.mat`, separar o sinal em janelas de 2.048 pontos e
  treinar somente com janelas normais. O teste pode então usar os rótulos
  conhecidos de falha.

## Glossário rápido

- **Rolamento:** conjunto de peças que permite ao eixo girar com menos atrito.
- **Vibração:** movimento pequeno e repetitivo do motor; falhas podem alterar
  seu padrão.
- **RPM:** rotações por minuto.
- **Feature:** número que resume uma propriedade de um trecho do sinal.
- **Autoencoder:** rede neural treinada para reconstruir entradas; aqui aprende
  principalmente o padrão normal.
- **Anomalia:** trecho cujo erro de reconstrução ultrapassa o limiar definido.
- **Falso positivo:** trecho normal sinalizado como falha.
- **Falso negativo:** falha que o modelo não sinaliza.

## Referências

- [Kaggle: CWRU Bearing Datasets](https://www.kaggle.com/datasets/brjapon/cwru-bearing-datasets/data)
- [Case Western Reserve University Bearing Data Center](https://engineering.case.edu/bearingdatacenter/welcome)