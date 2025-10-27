from collections import deque

def bfs(graph, start, goal):
    queue = deque([start])        # Cola de nodos por visitar
    visited = {start}             # Conjunto de nodos visitados
    parent = {start: None}        # Para reconstruir el camino

    while queue:
        node = queue.popleft()    # Saca el primer nodo de la cola
        print(f"Visitando: {node}")

        if node == goal:          # Si se llega al nodo meta
            path = []
            while node is not None:
                path.append(node)
                node = parent[node]
            return path[::-1]     # Devuelve el camino invertido

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = node
                queue.append(neighbor)

    return None                   # Si no hay camino


# Ejemplo de uso
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}

camino = bfs(graph, 'A', 'F')
print("\nCamino encontrado:", camino)
