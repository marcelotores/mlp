import numpy as np

#[[ 0.06290726  0.34951213]
# [-0.12585987  0.38149802]]


pesos = np.array([
    [0.48051112, -0.11836977, -0.25578928, -0.04523849,  0.02769548],
    [0.296582, 0.17906579, 0.20579868, 0.24060917, 0.18372743]
])
Yh = np.array([-0.26719822,  0.13125259,  0.12430141, -0.21210124, -0.17600597])
print(pesos.shape)
print(Yh.shape)
pesos = pesos * Yh
print(pesos)

#eh = np.dot(self.pesos_camada_2[1:, :], gradiente_Yo)
#gradiente
#[-0.07137476  0.30108423]

# Erro gerado pelo algoritmo
#-0.12585987  0.38149802

#print(pesos)