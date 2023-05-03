import numpy as np

S1 = np.array([
    [1, 2, 3],
    [1, 2, 3]
])

erro = np.array([
    [1, 2, 3],
    [1, 2, 3]
])

S2 = np.array([
    [1, 2, 3]
])

erro2 = np.array([
    [1, 2, 3]
])

print(np.dot(S1.T, erro))
print(np.dot(S2, erro2.T))