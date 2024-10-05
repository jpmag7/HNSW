import numpy as np
import random
import time


class HNSWNode():
    
    def __init__(self, vector, index, n_levels):
        self.vector = vector
        self.index = index
        self.neighbors = {i:[] for i in range(n_levels)}



class HNSWFlat():

    def __init__(self, points, n_levels, M, efConstruction=200):
        self.n_levels = n_levels
        self.M = M
        self.efConstruction = efConstruction
        self.entry_point = self._build_index(points, n_levels, M)



    def _find_nearest_neighbors(self, level, node, M):
        distances = [float("+inf") for _ in range(M)]
        neighbors = [None for _ in range(M)]
        size = len(level)-1
        
        for _ in range(self.efConstruction):
            neighbor = level[random.randint(0, size)]
            dist = np.linalg.norm(node.vector - neighbor.vector)
            m = max(distances)
            if dist < m:
                id = distances.index(m)
                distances[id] = dist
                neighbors[id] = neighbor
        
        return [n for n in neighbors if n != None]
    


    def _connect_neighbors(self, node, neighbors, level):
        for neighbor in neighbors:
            neighbor.neighbors[level].append(node)
            node.neighbors[level].append(neighbor)



    def _build_index(self, points, n_levels, M):
        data = [HNSWNode(p, i, n_levels) for i, p in enumerate(points)]
        
        decay = len(data) // n_levels
        for l in range(n_levels-1):
            for node in data:
                neighbors = self._find_nearest_neighbors(data, node, M)
                self._connect_neighbors(node, neighbors, l)
            l_size = len(data)
            for _ in range(decay):
                data.pop(random.randint(0, l_size-1))
                l_size -= 1
        entry_point = data[random.randint(0, len(data)-1)]
        return entry_point
    


    def _search_distance(self, query, values):
        dists = [np.linalg.norm(query - value.vector) for value in values]
        return dists
    


    def search(self, query):
        l = self.n_levels-1
        node = self.entry_point
        while l >= 0:
            dists = self._search_distance(query, [node] + node.neighbors[l])
            closer = dists.index(min(dists))

            while closer != 0:
                node = node.neighbors[l][closer-1]
                dists = self._search_distance(query, [node] + node.neighbors[l])
                closer = dists.index(min(dists))
            l -= 1
        return node.index, node.vector, dists[closer]
    

points = [np.random.rand(512) for _ in range(1000)]
query  = points[5]
index  = HNSWFlat(points, 8, 8)
print("Index built")
start = time.time()
idx, vect, d = index.search(query)
print(f"Position {idx}:\n\n{query}\n\n{vect}\n\n{idx}\n{d}")
print("Time:", time.time() - start)