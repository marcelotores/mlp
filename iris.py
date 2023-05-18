import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import seaborn
import mlp_p_amostra
from sklearn.datasets import load_iris

seaborn.set(style='whitegrid');
seaborn.set_context('talk')
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

iris_data = load_iris()

n_samples, n_features = iris_data.data.shape
def Show_Diagram(_x ,_y,title):
    plt.figure(figsize=(10,4))
    plt.scatter(iris_data.data[:,_x],
    iris_data.data[:, _y], c=iris_data.target, cmap=cm.viridis)
    plt.xlabel(iris_data.feature_names[_x]);
    plt.ylabel(iris_data.feature_names[_y]);
    plt.title(title)
    plt.colorbar(ticks=([0, 1, 2]));
    plt.show();
#Show_Diagram(0,1,'Sepal')
#Show_Diagram(2,3,'Petal')

random.seed(123)


def separate_data():
    part_A_train = iris_dataset[0:40]
    part_A_test = iris_dataset[40:50]
    part_B_train = iris_dataset[50:90]
    part_B_test = iris_dataset[90:100]
    part_C_train = iris_dataset[100:140]
    part_C_test = iris_dataset[140:150]
    train = np.concatenate((part_A_train,
                            part_B_train,
                            part_C_train))
    test = np.concatenate((part_A_test,
                           part_B_test,
                           part_C_test))
    return [train, test]


train_porc = 80  # Porcent Training
test_porc = 20  # Porcent Test
# Join X and Y
iris_dataset = np.column_stack((iris_data.data,
                                iris_data.target.T))
iris_dataset = list(iris_dataset)
random.shuffle(iris_dataset)
file_train, file_test = separate_data()
train_X = np.array([i[:4] for i in file_train])
train_y = np.array([i[4] for i in file_train])
test_X = np.array([i[:4] for i in file_test])
test_y = np.array([i[4] for i in file_test])

train_y = train_y.reshape((120, 1))
test_y = test_y.reshape((30, 1))


novo_y_treino = []
for i in train_y:
    if i == 0:
        novo_y_treino.append([1, -1, -1])
    elif i == 1:
        novo_y_treino.append([-1, 1, -1])
    else:
        novo_y_treino.append([-1, -1, 1])

y_treino_num = np.array(novo_y_treino)
#print(y_treino_num)
#print(test_y.shape)
#novo_y_teste = np.array([])
novo_y_teste = []
for i in test_y:
    if i == 0:
        novo_y_teste.append([1, -1, -1])
    elif i == 1:
        novo_y_teste.append([-1, 1, -1])
    else:
        novo_y_teste.append([-1, -1, 1])
y_num = np.array(novo_y_teste)
#print(y_num.shape)
#print(train_X)
#print(y_treino_num)

## Parâmetros
#taxa_aprendizado = 0.001
#epocas = 1
#qtd_neuronios_camada_oculta = 4
#qtd_neuronios_camada_saida = 3

## Definição de parâmetros
#mlp = mlpf.Mlp(train_X, taxa_aprendizado, epocas, qtd_neuronios_camada_oculta, qtd_neuronios_camada_saida)

## Treino
#errors, param, Z2 = mlp.treino(train_X, y_treino_num)


# ## Parâmetros
taxa_aprendizado = 0.1
epocas = 2
qtd_neuronios_camada_oculta = 4

qtd_neuronios_camada_saida = 3
#
## Definição de parâmetros
#
# Setando os pesos Iniciais
qtd_col_dataset = train_X.shape[1]
pesos_camada_1 = np.random.uniform(-0.5, 0.5, size=(qtd_col_dataset + 1,  qtd_neuronios_camada_oculta))
pesos_camada_2 = np.random.uniform(-0.5, 0.5, size=(qtd_neuronios_camada_oculta + 1, qtd_neuronios_camada_saida))
#
mlp = mlp_p_amostra.Mlp(train_X, taxa_aprendizado, epocas, qtd_neuronios_camada_oculta, qtd_neuronios_camada_saida, pesos_camada_1, pesos_camada_2)
#
## Treino
errors, parametros, p2, p1, eo_p = mlp.treino(train_X, y_treino_num)
y_predicao = mlp.predicao(test_X, parametros["pesos_camada_oculta"], parametros["pesos_camada_saida"])
print(y_predicao)

num_predicoes_corretas = (y_predicao == y_num).sum()
print('num_predicoes_corretas', num_predicoes_corretas)

#acuracia = (num_predicoes_corretas / y_num.shape[0]) * 100
#print('Acurácia: %.2f%%' % acuracia)


#print('####################################################################3')
#def tangente_hiperbolica(soma_dos_pesos):
#    """Função tangente hiperbólica."""

#    return (np.exp(soma_dos_pesos) - np.exp(-soma_dos_pesos)) / (np.exp(soma_dos_pesos) + np.exp(-soma_dos_pesos))
#print('######################### Z2 ###########################')
#print(Z2)
#print('######################### S2 ###########################')
#print(tangente_hiperbolica(Z2))


#xpoints = Z2[:10, 0]
#ypoints = tangente_hiperbolica(xpoints)

#print('soma pesos: ', xpoints)
#print('tamh: ', ypoints)
#plt.plot(xpoints, ypoints, 'o')
#plt.show()
#print(xpoints)
#print(ypoints)

## Teste
#y_predicao = mlp.predicao(test_X, param["pesos_camada_oculta"], param["pesos_camada_saida"], param["bias_camada_oculta"], param["bias_camada_saida"])
#print(y_predicao)
## Cálculo de acurácia
#num_predicoes_corretas = (y_predicao == y_num).sum()
#print('y_predicao', y_predicao)
#print('rotulos', y_num)
#print('num_predicoes_corretas', num_predicoes_corretas)

#acuracia = (num_predicoes_corretas / y_num.shape[0]) * 100
#print('Acurácia: %.2f%%' % acuracia)

##############

#print(errors)