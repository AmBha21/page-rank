import numpy as np

def page_ranking(adjaceny_matrix, d=0.85, tolerance=1e-6):
    col_sums = adjaceny_matrix.sum(axis=0)
    adjaceny_matrix[:, col_sums == 0] = 1 / adjaceny_matrix.shape[0]
    N = np.shape(adjaceny_matrix)[0]
    w = np.ones(N) / N
    teleport = np.ones(N) / N
    v = d * np.dot(adjaceny_matrix, w) + (1 - d) * teleport
    while np.linalg.norm(v - w) > tolerance:
        w = v
        v = d * np.dot(adjaceny_matrix, w) + (1 - d) * teleport
    return v

# example
adjaceny_matrix = np.array([
    [0,   0,   0,   0  ],
    [0,   0,   0,   0  ],
    [1, 0.5,   0,   0  ],
    [0, 0.5,   1,   0  ]
])

v = page_ranking(adjaceny_matrix)
print(v, sum(v))
    