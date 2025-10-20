import heapq

def uniform_cost_search(graph, start, goal):
    # Cola de prioridad con tuplas (costo acumulado, nodo actual)
    queue = [(0, start)]
    visited = set()
    parent = {start: None}
    cost_so_far = {start: 0}

    while queue:
        cost, node = heapq.heappop(queue)  # Extrae el nodo con menor costo
        print(f"Visitando: {node} con costo acumulado {cost}")

        if node == goal:  # Si llegamos al objetivo, reconstruimos el camino
            path = []
            while node is not None:
                path.append(node)
                node = parent[node]
            return path[::-1], cost  # Regresa el camino y su costo total

        if node in visited:
            continue
        visited.add(node)

        for neighbor, edge_cost in graph.get(node, []):
            new_cost = cost + edge_cost
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                parent[neighbor] = node
                heapq.heappush(queue, (new_cost, neighbor))

    return None, float("inf")  # Si no hay camino


# Ejemplo de uso
graph = {
    'A': [('B', 2), ('C', 5)],
    'B': [('D', 4), ('E', 1)],
    'C': [('F', 2)],
    'D': [],
    'E': [('F', 3)],
    'F': []
}

camino, costo_total = uniform_cost_search(graph, 'A', 'F')
print("\nCamino más barato encontrado:", camino)
print("Costo total del camino:", costo_total)
