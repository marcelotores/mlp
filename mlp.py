import numpy as np

class Mlp():
    def __init__(self, dataset, taxa_aprendizado=0.1, epocas=10, qtd_neuronios_camada_oculta=1, qtd_neuronios_camada_saida=1):

        self.taxa_aprendizado = taxa_aprendizado
        self.epocas = epocas
        self.qtd_neuronios_camada_oculta = qtd_neuronios_camada_oculta
        self.qtd_neuronios_camada_saida = qtd_neuronios_camada_saida

        qtd_col_dataset = dataset.shape[1]
        #choice(5, 3)

        self.pesos_camada_1 = np.random.uniform(-0.5, 0.5, size=(qtd_col_dataset + 1,  self.qtd_neuronios_camada_oculta))
        self.pesos_camada_2 = np.random.uniform(-0.5, 0.5, size=(self.qtd_neuronios_camada_oculta + 1, self.qtd_neuronios_camada_saida))

        print(f'######### Inicialialização dos pesos #########')
        print('Camada Oculta: ')
        print(self.pesos_camada_1[1:, :])
        print('Camdada de Saída')
        print(self.pesos_camada_2[1:, :])
        #print('bias: ', self.pesos_camada_1[0, :])

    def funcao_linear(self, pesos, dataset):
        return np.dot(dataset, pesos[1:, :]) + pesos[0, :]

    def sigmoide(self, soma_dos_pesos):
        return 1 / (1 + np.exp(-soma_dos_pesos))

    def step(self, pesos1):
        predicao = []
        for p1 in pesos1:
            if p1 > 0:
                predicao.append(1)
            else:
                predicao.append(0)
        arr = np.array(predicao)
        return arr

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
        return np.where(S2 >= 0.5, 1, 0)
        #return np.where(S2 <= -0.6, -1, np.where(S2 <= 0.6, 0, 1))
        #return np.where(S2 <= 0.33, -1, np.where(S2 <= 0.66, 0, 1))

    def treino(self, X, y):
        ## ~~ Initialize parameters ~~##

        ## ~~ storage errors after each iteration ~~##
        errors = []
        print('##################### Treino #####################')

        for _ in range(self.epocas):
            print(f'############ Época {_} ############')

            ## Forward ##

            Z1 = self.funcao_linear(self.pesos_camada_1, X)
            S1 = self.tangente_hiperbolica(Z1)
            #S1 = self.sigmoide(Z1)
            Z2 = self.funcao_linear(self.pesos_camada_2, S1)
            S2 = self.tangente_hiperbolica(Z2)
            #S2 = self.sigmoide(Z2)

            ## Erros ##
            #error = self.custo(S2, y)
            #errors.append(error)
            #print('Z1', Z2)
            #print('S2', S2)
            erro_camada_saida = S2 - y

            #print('Erro camada saida', erro_camada_saida)


            derivada2 = (S2 * (1 - S2))
            derivada1 = (S1 * (1 - S1))

            # Nessa forma, os neuronios são as linhas e as amostras, as colunas
            #print(np.dot(self.pesos_camada_2[1:, :], erro_camada_saida.T))

            # Oposto
            #print(np.dot(self.pesos_camada_2[1:, :], erro_camada_saida.T).T)
            erro_camada_oculta  = np.dot(self.pesos_camada_2[1:, :], erro_camada_saida.T).T

            gradiente1 = erro_camada_oculta * derivada1
            gradiente2 = erro_camada_saida * derivada2

            print('gradiente 1:', gradiente1)
            print('gradiente 2:', gradiente2)
            print('S1:', S1)


            #gradiente

            ## Calcula os Gradientes ##

            # gradiente de saída
            #delta2 = (S2 - y) * (S2 * (1 - S2))

            #gradiente_peso2 = np.dot(S1.T, delta2)

            #db2 = np.sum(delta2, axis=0)

            # gradiente da camada oculta
            #delta1 = np.dot(delta2, self.pesos_camada_2[1:, :].T) * (S1 * (1 - S1))
            #gradiente_peso1 = np.dot(X.T, delta1)
            #db1 = np.sum(delta1, axis=0)

            # Atualização dos pesos
            ##self.pesos_camada_2 -= self.taxa_aprendizado * gradiente_peso2 * S2
            self.pesos_camada_2[1:, :] -= self.taxa_aprendizado * gradiente2 * S2
            self.pesos_camada_1[1:, :] -= self.taxa_aprendizado * gradiente1

            #print('Z2', Z2)
            #print('S2', S2)


            parametros = {
                "pesos_camada_oculta": self.pesos_camada_1,
                "pesos_camada_saida": self.pesos_camada_2
            }

        return errors, parametros
