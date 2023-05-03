import numpy as np
from mlp import Mlp

################################## Xor ####################################

## Rótulos
y = np.array([[0, 1, 1, 0]]).T


## Dataset

X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]]).T
print('X', X.shape)
## Parâmetros
taxa_aprendizado = 0.1
epocas = 1
qtd_neuronios_camada_oculta = 4

qtd_neuronios_camada_saida = 3

## Definição de parâmetros
mlp = Mlp(X, taxa_aprendizado, epocas, qtd_neuronios_camada_oculta, qtd_neuronios_camada_saida)

## Treino
errors, param = mlp.treino(X, y)
