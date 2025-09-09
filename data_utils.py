# data_utils.py
import numpy as np
import math
from itertools import permutations

def generate_euclidean_instances(batch_size, n_nodes, seed=None):
    if seed is not None:
        np.random.seed(seed)
    # coords in [0,1]
    coords = np.random.rand(batch_size, n_nodes, 2).astype(np.float32)
    return coords

def pairwise_euclidean_distance(coords):
    # coords: (..., n, 2)
    diff = coords[:, :, None, :] - coords[:, None, :, :]  # (B, n, n, 2)
    d = np.linalg.norm(diff, axis=-1)  # (B, n, n)
    return d

def tour_length(coords, tour):
    # coords shape (n,2); tour is a permutation list/array length n
    c = coords[tour]
    # close loop
    c_next = np.roll(c, -1, axis=0)
    return np.linalg.norm(c - c_next, axis=1).sum()

# Small exact solver (Held-Karp brute-force for small n)
def exact_tsp_solution(coords):
    # coords: (n,2)
    n = coords.shape[0]
    best = None
    best_len = float('inf')
    for perm in permutations(range(n)):
        L = tour_length(coords, perm)
        if L < best_len:
            best_len = L
            best = perm
    return list(best), best_len
