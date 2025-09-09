# baselines.py
import numpy as np

def nearest_neighbor(coords):
    # coords: (n,2)
    n = coords.shape[0]
    visited = [0]
    unvisited = set(range(1, n))
    while unvisited:
        last = visited[-1]
        next_node = min(unvisited, key=lambda x: np.linalg.norm(coords[last]-coords[x]))
        visited.append(next_node)
        unvisited.remove(next_node)
    return visited

def two_opt_swap(tour, i, k):
    new_tour = tour[:i] + tour[i:k+1][::-1] + tour[k+1:]
    return new_tour

def two_opt(coords, tour):
    n = len(tour)
    improved = True
    best = tour[:]
    best_len = compute_length(coords, best)
    while improved:
        improved = False
        for i in range(1, n-1):
            for k in range(i+1, n):
                new_tour = two_opt_swap(best, i, k)
                L = compute_length(coords, new_tour)
                if L < best_len - 1e-9:
                    best = new_tour
                    best_len = L
                    improved = True
                    break
            if improved:
                break
    return best

def compute_length(coords, tour):
    coords_t = coords[tour]
    nexts = np.roll(coords_t, -1, axis=0)
    return np.linalg.norm(coords_t - nexts, axis=1).sum()
