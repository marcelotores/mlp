import numpy as np

class Mlp():
    def __init__(self, dataset, taxa_aprendizado=0.1, epocas=10, qtd_neuronios_camada_oculta=1, qtd_neuronios_camada_saida=1, pesos_camada_1=None, pesos_camada_2=None):

        self.taxa_aprendizado = taxa_aprendizado
        self.epocas = epocas
        self.qtd_neuronios_camada_oculta = qtd_neuronios_camada_oculta
        self.qtd_neuronios_camada_saida = qtd_neuronios_camada_saida

        qtd_col_dataset = dataset.shape[1]
        #choice(5, 3)
        self.pesos_camada_1 = pesos_camada_1
        self.pesos_camada_2 = pesos_camada_2
        #self.pesos_camada_1 = np.random.uniform(-0.5, 0.5, size=(qtd_col_dataset + 1,  self.qtd_neuronios_camada_oculta))
        #self.pesos_camada_2 = np.random.uniform(-0.5, 0.5, size=(self.qtd_neuronios_camada_oculta + 1, self.qtd_neuronios_camada_saida))

        print(f'######### Inicialialização dos pesos #########')
        print('Camada Oculta: ')
        print('Pesos: ')
        print(self.pesos_camada_1[1:, :])
        print('Bias: ')
        print(self.pesos_camada_1[:1, ])

        print('Camdada de Saída')
        print('Pesos: ')
        print(self.pesos_camada_2[1:, :])
        print('Bias: ')
        print(self.pesos_camada_2[:1, ])


    def funcao_linear(self, pesos, dataset):
        return np.dot(dataset, pesos[1:, :]) + pesos[0, :]

    def sigmoide(self, soma_dos_pesos):
        return 1 / (1 + np.exp(-soma_dos_pesos))

    def tangente_hiperbolica(self, soma_dos_pesos):
        """Função tangente hiperbólica."""

        valor = (np.exp(soma_dos_pesos) - np.exp(-soma_dos_pesos)) / (np.exp(soma_dos_pesos) + np.exp(-soma_dos_pesos))
        return valor

    def custo(self, neuronios_ativados, rotulos):
        return (np.mean(np.power(neuronios_ativados - rotulos, 2))) / 2

    def predicao(self, dataset, pesos_camada_1, pesos_camada_2):

        print('##################### Teste #####################')

        Z1 = self.funcao_linear(pesos_camada_1, dataset)
        #S1 = self.step(Z1)
        #S1 = self.sigmoide(Z1)
        S1 = self.tangente_hiperbolica(Z1)
        Z2 = self.funcao_linear(pesos_camada_2, S1)
        #S2 = self.step(Z2)
        S2 = self.tangente_hiperbolica(Z2)
        #S2 = self.sigmoide(Z2)
        #return np.where(S2 >= 0.5, 1, 0)
        #return np.where(S2 <= -0.6, -1, np.where(S2 <= 0.6, 0, 1))
        return np.where(S2 <= 0.33, -1, np.where(S2 <= 0.66, 0, 1))

    def treino(self, X, y):
        ## ~~ Initialize parameters ~~##

        ## ~~ storage errors after each iteration ~~##
        errors = []
        print('##################### Treino #####################')
        p2 = np.array([])
        p1 = np.array([])
        eo_p = np.array([])

        for _ in range(self.epocas):
            print(f'############ Época {_} ############')

            for input, target in zip(X, y):

                # IDA
                Z1 = self.funcao_linear(self.pesos_camada_1, input)
                #Yh = self.tangente_hiperbolica(Z1)
                Yh = self.sigmoide(Z1)
                Z2 = self.funcao_linear(self.pesos_camada_2, Yh)
                Yo = self.sigmoide(Z2)
                #Yo = self.tangente_hiperbolica(Z2)

                # Erro camada saída
                eo = target - Yo
                eo_p = np.append(eo_p, eo)

                # Derivada do Valor previsto da camada de saída
                derivada_Yo = ((1 - Yo) ** 2) / 2
                #derivada_Yo = Yo * (1 - Yo)

                # Gradientes locais para cada neuronio da camada de saída
                gradiente_Yo = derivada_Yo * eo

                #Erro camada oculta
                eh = np.dot(self.pesos_camada_2[1:, :], gradiente_Yo)

                # Derivada do valor previsto da camada oculta
                #derivada_Yh = ((1 - Yh) ** 2) / 2
                derivada_Yh = Yh * (1 - Yh)
                # Gradientes locais para cada neuronio da camada oculta
                gradiente_Yh = derivada_Yh * eh

                ## Atualização dos pesos

                self.pesos_camada_2[1:, :] += self.taxa_aprendizado * gradiente_Yo
                novos_pesos_2 = self.pesos_camada_2[1:, :].T * Yh
                self.pesos_camada_2[1:, :] = novos_pesos_2.T
                self.pesos_camada_1[1:, :] += self.taxa_aprendizado * gradiente_Yh

                #self.pesos_camada_2[0, :] += gradiente_Yo
                #self.pesos_camada_1[0, :] += gradiente_Yh

                print(f'Entrada={input}, ground-truth={target}, pred={np.where(Yo <= 0.33, -1, np.where(Yo <= 0.66, 0, 1))}')
                p2 = np.append(p2, self.pesos_camada_2[1, :])
                p1 = np.append(p1, self.pesos_camada_2[2, :])


                #p1 = np.append(p1, print(self.pesos_camada_2[1:, :]))
                #p1 = np.append(p1, self.pesos_camada_2[1, :][1])
                #p2.append(self.pesos_camada_2[1, :][0])
                #print(self.pesos_camada_2[0, :])
                #print(pesos_camada_2[0, :])
                #print('----------- ', input, target, '----------------')

                #print('Z1', Z1)
                #print('Yh', Yh)
                #print('Z2', Z2)
                #print('Yo', Yo)
                #print('eo: ', eo)
                #print('derivada_Yo: ', derivada_Yo)
                #print('gradiente_Yo', gradiente_Yo)
                #print('----')
                #print('Pesos Ocultos: ')
                #print(self.pesos_camada_1[1:, :])
                #print(self.pesos_camada_2[1:, :])
                #print('Pesos Saída: ')




            parametros = {
                "pesos_camada_oculta": self.pesos_camada_1,
                "pesos_camada_saida": self.pesos_camada_2
            }

        return errors, parametros, p2, p1, eo_p


# ## Rótulos
y = np.array([[-1, 1, 1, -1]]).T
y_and = np.array([[0, 0, 0, 1]]).T
X = np.array([[-1, -1, 1, 1],
               [-1, 1, -1, 1]]).T
# print('X', X.shape)
# ## Parâmetros
taxa_aprendizado = 0.1
epocas = 5
qtd_neuronios_camada_oculta = 2

qtd_neuronios_camada_saida = 1
#
## Definição de parâmetros
#
# Setando os pesos Iniciais
qtd_col_dataset = X.shape[1]
pesos_camada_1 = np.random.uniform(-0.5, 0.5, size=(qtd_col_dataset + 1,  qtd_neuronios_camada_oculta))
pesos_camada_2 = np.random.uniform(-0.5, 0.5, size=(qtd_neuronios_camada_oculta + 1, qtd_neuronios_camada_saida))
#
mlp = Mlp(X, taxa_aprendizado, epocas, qtd_neuronios_camada_oculta, qtd_neuronios_camada_saida, pesos_camada_1, pesos_camada_2)
#
## Treino
#########errors, parametros, p2, p1, eo_p = mlp.treino(X, y)
#print(p2)
import matplotlib.pyplot as plt
#xpoints = np.array([0, 10])
#####ypoints = np.array(p2)
#####yypoints = np.array(p1)
#####erro = np.array(eo_p)
#####print(erro)
#####plt.plot(erro)
#plt.plot(ypoints)
#plt.plot(yypoints)
#####plt.show()
