import numpy as np

def page_ranking(stochastic_matrix, d=0.85, tolerance=1e-6):
    # getting the number of "websites"
    N = np.shape(stochastic_matrix)[0]

    # dealing with axis that have a sum of 0, i.e. no outgoing links
    col_sum = stochastic_matrix.sum(axis=0)
    stochastic_matrix[:, col_sum == 0] = 1 / N

    # initializing pagerank for each website
    w = np.ones(N) / N

    # initializing teleportation vector
    teleport = np.ones(N) / N
    
    v = d * np.dot(stochastic_matrix, w) + (1 - d) * teleport

    iterations = 0
    # iterating until convergence
    while np.linalg.norm(v - w) > tolerance:
        w = v
        # using the given pagerank formula
        v = d * np.dot(stochastic_matrix, w) + (1 - d) * teleport

        iterations += 1
        print(v, iterations)
    return v, iterations

# example
stochastic_matrix = np.array([
    [0,   0,   0,   0.25  ],
    [0,   0,   0,   0.25  ],
    [1, 0.5,   0,   0.5  ],
    [0, 0.5,   1,   0  ]
])

# stochastic_matrix = np.array([
#     [0,   0,   0,   0  ],
#     [0,   0,   0,   0  ],
#     [1, 0.5,   0,   0  ],
#     [0, 0.5,   1,   0  ]
# ])

v, iterations = page_ranking(stochastic_matrix)
print(v, sum(v), iterations)
    