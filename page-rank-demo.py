import numpy as np

def page_ranking(adjaceny_matrix, d=0.85, tolerance=1e-6):
    # getting the number of "websites"
    N = np.shape(adjaceny_matrix)[0]

    # initializing pagerank for each website
    w = np.ones(N) / N

    # initializing teleportation vector
    teleport = np.ones(N) / N
    v = d * np.dot(adjaceny_matrix, w) + (1 - d) * teleport

    iterations = 0
    # iterating until convergence
    while np.linalg.norm(v - w) > tolerance:
        w = v
        # using the given pagerank formula
        v = d * np.dot(adjaceny_matrix, w) + (1 - d) * teleport

        iterations += 1
    return v, iterations

# example
adjaceny_matrix = np.array([
    [0,   0,   0,   0.25  ],
    [0,   0,   0,   0.25  ],
    [1, 0.5,   0,   0.5  ],
    [0, 0.5,   1,   0  ]
])

v, iterations = page_ranking(adjaceny_matrix)
print(v, sum(v), iterations)
    